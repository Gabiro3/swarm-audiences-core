"""Phase-1 cheap filters: YOLO (visual) + Whisper (text) + CLAP/FAISS (audio).

Ported from the notebook's in-process `frontrunner_triage()` (Cells 2/3/7-8),
with two production fixes:
  - per-request unique temp wav files instead of a single shared "fr_audio.wav"
    (the notebook processed one clip at a time interactively; the API serves
    concurrent requests)
  - all GPU-touching work serialized behind `locks.inference_lock`
"""

import bisect
import math
import os
import re
import subprocess
import tempfile
import threading
import uuid

import numpy as np
import torch
import faiss
from ultralytics import YOLO
from faster_whisper import WhisperModel
from transformers import AutoProcessor, AutoModel

from . import config
from .locks import inference_lock
from .media import has_video_stream

_fr: dict = {}
_load_lock = threading.Lock()


def is_ready() -> bool:
    return bool(_fr.get("ready"))


def load_frontrunner(device: str = config.FRONTRUNNER_DEVICE) -> dict:
    if _fr.get("ready"):
        return _fr
    with _load_lock:
        if _fr.get("ready"):
            return _fr
        dev = device if torch.cuda.is_available() else "cpu"
        _fr["device"] = dev

        # Absolute path, not a bare "yolov8n.pt" — that downloads into the
        # process's current working directory, which isn't guaranteed to be
        # writable (e.g. /app in the container image) or persisted.
        os.makedirs(os.path.dirname(config.YOLO_WEIGHTS_PATH), exist_ok=True)
        _fr["yolo"] = YOLO(config.YOLO_WEIGHTS_PATH)

        if dev == "cuda":
            cap = torch.cuda.get_device_capability(0)
            major = cap[0] if isinstance(cap, tuple) else cap
            compute_type = "float16" if major >= 7 else "int8"
        else:
            compute_type = "int8"
        _fr["whisper"] = WhisperModel("base", device=dev, compute_type=compute_type)

        _fr["clap_proc"] = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        _fr["clap_model"] = AutoModel.from_pretrained("laion/clap-htsat-unfused").to(dev)

        txt = _fr["clap_proc"](text=config.DANGER_PHRASES, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = _fr["clap_model"].get_text_features(**txt.to(dev))
            feats = getattr(out, "text_embeds", None)
            if feats is None:
                feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = feats / feats.norm(dim=-1, keepdim=True)
            seed = feats.cpu().numpy().astype("float32")
        idx = faiss.IndexFlatIP(seed.shape[1])
        idx.add(seed)
        _fr["faiss"] = idx

        _fr["ready"] = True
    return _fr


def _extract_audio(video_path: str):
    wav = os.path.join(tempfile.gettempdir(), f"fr_audio_{uuid.uuid4().hex}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "48000", "-ac", "1", wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if os.path.exists(wav) and os.path.getsize(wav) > 1024:
        return wav
    if os.path.exists(wav):
        os.remove(wav)
    return None


def _visual_track(video_path: str) -> float:
    if not has_video_stream(video_path):  # audio-only files have no frames
        return 0.0
    # c_v = max confidence among DANGER-class detections only (not any COCO object).
    results = _fr["yolo"].predict(source=video_path, vid_stride=30, verbose=False)
    c_v = 0.0
    for r in results:
        if r.boxes is None:
            continue
        for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if r.names[int(cls_id)] in config.DANGER_CLASSES:
                c_v = max(c_v, float(conf))
    return c_v


def _text_track(wav):
    if not wav:
        return 0.0, ""
    segments, _ = _fr["whisper"].transcribe(wav, beam_size=5)
    transcript = " ".join(seg.text for seg in segments).lower()
    # word-boundary match so "skills" doesn't trip "kill"
    hits = sum(1 for w in config.BANNED_LEXICON if re.search(rf"\b{re.escape(w)}\b", transcript))
    return min(1.0, hits * 0.25), transcript


def _audio_track(wav):
    if not wav:
        return 0.0, "None"
    import librosa
    audio_data, _ = librosa.load(wav, sr=48000)
    proc, dev = _fr["clap_proc"], _fr["device"]
    inputs = proc(audio=audio_data, sampling_rate=48000, return_tensors="pt")
    with torch.no_grad():
        out = _fr["clap_model"].get_audio_features(**inputs.to(dev))
        feats = getattr(out, "audio_embeds", None)
        if feats is None:
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        emb = feats.cpu().numpy().astype("float32")
    dist, ind = _fr["faiss"].search(emb, k=1)
    c_a = max(0.0, float(dist[0][0]))  # clamp: cosine can be negative, keep score in [0,1]
    return c_a, config.DANGER_PHRASES[int(ind[0][0])]


def _route(gate_tracks, raw_tracks):
    """gate_tracks drive r_total/discrepancy (calibrated, when fitted); raw_tracks
    are the uncalibrated scores used for the vlm_confirms fast-kill so a bad
    calibration fit can never fabricate a block on its own.
    """
    g = np.array(gate_tracks, dtype=float)
    rw = np.array(raw_tracks, dtype=float)
    r_total = config.ALPHA * g.max() + (1 - config.ALPHA) * np.sqrt(np.mean(g ** 2))
    disc = float(np.var(g))
    s_a = min(1.0, float(r_total * (1.0 + config.GAMMA * disc)))  # combined score (reported)

    if config.PASS_POLICY == "vlm_confirms":
        # the cheap funnel may ONLY fast-kill near-certain violations (on the RAW
        # signal); everything else escalates — Qwen decides PUBLISH, not the funnel.
        verdict = "AUTO-BLOCK" if float(rw.max()) >= config.HARD_BLOCK_AT else "GRAY_ZONE_ESCALATE"
    elif config.ROUTING == "legacy_product":
        verdict = "GRAY_ZONE_ESCALATE"
        if s_a < config.LEGACY_PASS:
            verdict = "AUTO-PASS"
        elif s_a >= config.LEGACY_BLOCK:
            verdict = "AUTO-BLOCK"
    else:  # discrepancy_gated — modalities disagree => send to the VLM
        if disc >= config.DISC_GATE:
            verdict = "GRAY_ZONE_ESCALATE"
        elif r_total < config.PASS_AT:
            verdict = "AUTO-PASS"
        elif r_total >= config.BLOCK_AT:
            verdict = "AUTO-BLOCK"
        else:
            verdict = "GRAY_ZONE_ESCALATE"
    return verdict, r_total, disc, s_a


def _raw_tracks(video_path: str) -> dict:
    wav = _extract_audio(video_path)
    has_audio = wav is not None
    try:
        c_v = _visual_track(video_path)
        c_t, transcript = _text_track(wav)
        c_a, matched = _audio_track(wav)
    finally:
        if wav and os.path.exists(wav):
            os.remove(wav)
    return {
        "visual": c_v, "text": c_t, "audio": c_a,
        "transcript": transcript, "acoustic_match": matched, "has_audio": has_audio,
    }


class TrackCalibrator:
    """Rescales each modality (robust_minmax / ecdf / zlogistic) before routing."""

    TRACKS = ("visual", "text", "audio")

    def __init__(self, method: str = "robust_minmax", p_low: float = 5, p_high: float = 95,
                 min_samples: int = 20):
        self.method, self.p_low, self.p_high, self.min_samples = method, p_low, p_high, min_samples
        self.fitted = False
        self.lohi: dict = {}
        self.samples = {t: np.array([0.0]) for t in self.TRACKS}

    def fit(self, rows):
        # A bad fit on too few samples can collapse a real signal toward 0 and
        # cause a false clear — refuse rather than fit on a thin sample.
        if len(rows) < self.min_samples:
            self.fitted = False
            return self
        for t in self.TRACKS:
            vals = np.array([float(r[t]) for r in rows], dtype=float)
            self.samples[t] = vals
            self.lohi[t] = (float(np.percentile(vals, self.p_low)),
                             float(np.percentile(vals, self.p_high)))
        self.fitted = True
        return self

    def transform(self, t: str, x: float) -> float:
        if not self.fitted:
            return x
        if self.method == "robust_minmax":
            lo, hi = self.lohi[t]
            if hi - lo < 1e-6:  # degenerate track (e.g. all-zero sample) — pass raw
                return float(x)  # through; never fabricate a 0 that could mask a real signal
            return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))
        if self.method == "ecdf":
            s = sorted(self.samples[t].tolist())
            return bisect.bisect_left(s, x) / len(s) if s else x
        if self.method == "zlogistic":
            vals = self.samples[t]
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = max(float(q3 - q1), 1e-6)
            return 1.0 / (1.0 + math.exp(-((x - med) / iqr)))
        return x

    def transform_triple(self, cv: float, ct: float, ca: float):
        return (self.transform("visual", cv), self.transform("text", ct), self.transform("audio", ca))


CALIBRATOR = TrackCalibrator(method=config.CALIBRATION_METHOD, min_samples=config.MIN_CALIB_SAMPLES)


def fit_calibration(sample_paths, method=None, p_low=5, p_high=95, min_samples=None):
    """Sample raw track scores over clips, fit per-track rescaling, install globally.

    Fit ONLY from a held-out background set of known-benign clips — never the
    clips you're about to judge — and only if there are enough samples; below
    `min_samples` the fit is refused and routing stays on raw scores (safe).
    """
    global CALIBRATOR
    with inference_lock:
        load_frontrunner()
        calibrator = TrackCalibrator(
            method=method or config.CALIBRATION_METHOD, p_low=p_low, p_high=p_high,
            min_samples=min_samples if min_samples is not None else config.MIN_CALIB_SAMPLES,
        )
        rows, errors = [], []
        for p in sample_paths:
            try:
                rows.append(_raw_tracks(p))
            except Exception as e:
                errors.append({"path": os.path.basename(p), "error": type(e).__name__})
        calibrator.fit(rows)
        CALIBRATOR = calibrator
    return calibrator, rows, errors


def frontrunner_triage(video_path: str) -> dict:
    with inference_lock:
        load_frontrunner()
        rt = _raw_tracks(video_path)
        c_v, c_t, c_a = rt["visual"], rt["text"], rt["audio"]
        if CALIBRATOR.fitted:
            cv, ct, ca = CALIBRATOR.transform_triple(c_v, c_t, c_a)
            routed_on = "calibrated"
        else:
            cv, ct, ca = c_v, c_t, c_a
            routed_on = "raw"
        verdict, r_total, disc, s_a = _route((cv, ct, ca), (c_v, c_t, c_a))

    return {
        "verdict": verdict,
        "adjusted_risk_score": round(s_a, 4),
        "r_total": round(float(r_total), 4),
        "discrepancy": round(disc, 4),
        "metrics": {"visual": round(c_v, 4), "text": round(c_t, 4), "audio": round(c_a, 4)},
        "metrics_calibrated": (
            {"visual": round(cv, 4), "text": round(ct, 4), "audio": round(ca, 4)}
            if routed_on == "calibrated" else None
        ),
        "routed_on": routed_on,
        "acoustic_match": rt["acoustic_match"],
        "extracted_transcript": rt["transcript"],
        "has_audio": rt["has_audio"],
    }
