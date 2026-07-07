"""Dataset adapters for benchmark_runner.py.

Each adapter is a generator that yields `ClipRef` — a normalized pointer to
one clip plus whatever ground-truth label the dataset provides (or
"unlabeled" if it doesn't have one, e.g. OpenVid/WebVid are caption datasets,
not violence/safety-labeled). benchmark_runner.py doesn't care which adapter
produced a ClipRef; it just submits it to the API.

None of these datasets ship in a single universal format, and this script
can't reach the internet to inspect your actual local copy, so treat the
dataset-specific adapters below as best-effort starting points:

  - webvid            : official CSV ships a direct `contentUrl` -> no download
                         needed, referenced by URL. Solid out of the box.
  - openvid           : official manifest CSV lists filenames, not URLs — the
                         actual .mp4s come from separate HF-hosted archives you
                         must already have extracted locally.
  - ucf_crime (labeled_dir) : matches the official release's `<Category>/<clip>.mp4`
                         folder layout.
  - xd_violence       : XD-Violence encodes labels in the filename itself
                         (community convention, e.g. "..._label_B2-0-0.mp4").
                         The LABEL_MAP below may not match every mirror of the
                         dataset — verify a few rows before trusting it.
  - hdvila (HD-VILA-100M) : ships as YouTube video-id + clip-span metadata, not
                         downloadable video files. This adapter expects clips
                         you've already extracted locally (or can optionally
                         shell out to yt-dlp+ffmpeg to fetch them — slow, and
                         subject to YouTube's terms, so it's opt-in).

If none of these match your local layout, use `iter_manifest_csv` — point it
at a CSV you generate yourself with columns (video_id, ground_truth, path_or_url)
and it works for any dataset.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


@dataclass
class ClipRef:
    video_id: str
    source: str
    ground_truth: str
    file_path: Optional[str] = None
    video_url: Optional[str] = None

    def __post_init__(self):
        if bool(self.file_path) == bool(self.video_url):
            raise ValueError(f"ClipRef {self.video_id!r} needs exactly one of file_path/video_url")


# ---------------------------------------------------------------------------
# Generic escape hatch — works for any dataset once you have a manifest CSV
# ---------------------------------------------------------------------------

def iter_manifest_csv(
    csv_path: str,
    source: str,
    *,
    id_col: str = "video_id",
    label_col: str = "ground_truth",
    path_col: str = "path",
    url_col: str = "url",
) -> Iterator[ClipRef]:
    """Read a manifest with columns video_id, ground_truth, and EITHER path OR url
    (whichever is present per row). Use this for any dataset that doesn't fit
    the more specific adapters below."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            path = (row.get(path_col) or "").strip()
            url = (row.get(url_col) or "").strip()
            yield ClipRef(
                video_id=(row.get(id_col) or "").strip() or f"{source}_{i}",
                source=source,
                ground_truth=(row.get(label_col) or "unlabeled").strip(),
                file_path=path or None,
                video_url=url or None,
            )


# ---------------------------------------------------------------------------
# WebVid-10M — official CSV has a direct `contentUrl`, no local download needed
# ---------------------------------------------------------------------------

def iter_webvid(csv_path: str, source: str = "webvid") -> Iterator[ClipRef]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            video_id = row.get("videoid") or row.get("video_id")
            url = row.get("contentUrl") or row.get("content_url")
            if not video_id or not url:
                continue
            yield ClipRef(video_id=str(video_id), source=source, ground_truth="unlabeled", video_url=url)


# ---------------------------------------------------------------------------
# OpenVid-1M — manifest CSV lists filenames; videos must already be extracted
# locally from the HF-hosted archive parts into `video_dir`.
# ---------------------------------------------------------------------------

def iter_openvid(csv_path: str, video_dir: str, source: str = "openvid") -> Iterator[ClipRef]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            filename = row.get("video") or row.get("video_path") or row.get("filename")
            if not filename:
                continue
            local_path = os.path.join(video_dir, filename)
            if not os.path.isfile(local_path):
                continue  # not extracted locally yet — skip rather than fail the whole run
            yield ClipRef(
                video_id=os.path.splitext(filename)[0],
                source=source,
                ground_truth="unlabeled",
                file_path=local_path,
            )


# ---------------------------------------------------------------------------
# Generic labeled-folder walker — matches UCF-Crime's official
# `<root>/<Category>/<clip>.mp4` layout (Category folders double as labels,
# including things like "Testing_Normal_Videos_Anomaly" -> normalized below).
# ---------------------------------------------------------------------------

def iter_labeled_dir(root_dir: str, source: str) -> Iterator[ClipRef]:
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        label = os.path.basename(dirpath)
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            path = os.path.join(dirpath, name)
            gt = "Normal" if "normal" in label.lower() else label
            yield ClipRef(video_id=os.path.splitext(name)[0], source=source, ground_truth=gt, file_path=path)


# ---------------------------------------------------------------------------
# XD-Violence — labels are embedded in the filename itself (community
# convention). Verify LABEL_MAP against a sample of your actual filenames;
# mirrors of this dataset are not perfectly consistent.
# ---------------------------------------------------------------------------

XD_VIOLENCE_LABEL_MAP = {
    "A": "Normal",
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "CarAccident",
    "G": "Explosion",
}


def _parse_xd_violence_label(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    marker = "_label_"
    if marker not in stem:
        return "unknown"
    token = stem.split(marker, 1)[1]
    token = token.split("-")[0]  # strip the trailing "-0-0" segment index
    parts = [p for p in token.split("&") if p]
    mapped = [XD_VIOLENCE_LABEL_MAP.get(p, p) for p in parts] or ["unknown"]
    return "+".join(mapped)


def iter_xd_violence(root_dir: str, source: str = "xd_violence") -> Iterator[ClipRef]:
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            yield ClipRef(
                video_id=os.path.splitext(name)[0],
                source=source,
                ground_truth=_parse_xd_violence_label(name),
                file_path=os.path.join(dirpath, name),
            )


# ---------------------------------------------------------------------------
# HD-VILA-100M — metadata-only dataset (YouTube video id + clip span); real
# video bytes require a separate yt-dlp+ffmpeg extraction step. This adapter
# looks for clips you've already extracted; set fetch_missing=True to have it
# shell out to yt-dlp/ffmpeg on the fly for anything missing (slow, subject to
# YouTube's terms of service — off by default).
# ---------------------------------------------------------------------------

def _ytdlp_fetch_clip(youtube_url: str, span: list, out_path: str, timeout_s: int = 300) -> bool:
    if shutil.which("yt-dlp") is None or shutil.which("ffmpeg") is None:
        return False
    tmp_path = out_path + ".full.mp4"
    try:
        subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", tmp_path, youtube_url],
            check=True, timeout=timeout_s, capture_output=True,
        )
        start, end = span[0], span[1]
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", tmp_path,
             "-c", "copy", out_path],
            check=True, timeout=timeout_s, capture_output=True,
        )
        return os.path.isfile(out_path)
    except Exception:
        return False
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)


def iter_hdvila(
    jsonl_path: str,
    clips_dir: str,
    source: str = "hdvila",
    fetch_missing: bool = False,
) -> Iterator[ClipRef]:
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            clip_id = rec.get("clip_id") or rec.get("video_id")
            if not clip_id:
                continue
            local_path = rec.get("clip_path") or os.path.join(clips_dir, f"{clip_id}.mp4")
            if not os.path.isfile(local_path):
                if fetch_missing and rec.get("url") and rec.get("span"):
                    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                    if not _ytdlp_fetch_clip(rec["url"], rec["span"], local_path):
                        continue
                else:
                    continue  # not extracted locally — skip (see module docstring)
            yield ClipRef(video_id=str(clip_id), source=source, ground_truth="unlabeled", file_path=local_path)
