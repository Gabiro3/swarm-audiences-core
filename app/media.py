"""Shared video helpers used by both the frontrunner and the Qwen-VL auditor."""

import base64
import glob
import os
import subprocess
import tempfile

import cv2


def has_video_stream(path: str) -> bool:
    """ffprobe is the authoritative check — handles AV1/HEVC/VP9 that OpenCV may reject."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15,
        )
        if "video" in r.stdout:
            return True
    except Exception:
        pass
    # OpenCV fallback (sequential read avoids the seek-based hardware-decode path)
    cap = cv2.VideoCapture(path)
    try:
        ok, _ = cap.read()
        return bool(ok)
    except Exception:
        return False
    finally:
        cap.release()


def sample_frames_base64(
    path: str,
    target_fps: float = 1.0,
    max_frames: int = 16,
    jpeg_quality: int = 80,
) -> list[str]:
    """Evenly sample frames at ~target_fps (capped to max_frames).

    Uses ffmpeg subprocess first — it handles AV1, HEVC, VP9, and any other
    codec via its software decoders. OpenCV is kept as a fallback only.
    Returns a list of "data:image/jpeg;base64,..." URIs.
    """
    frames = _sample_via_ffmpeg(path, target_fps, max_frames, jpeg_quality)
    if not frames:
        frames = _sample_via_opencv(path, target_fps, max_frames, jpeg_quality)
    return frames


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_via_ffmpeg(path: str, target_fps: float, max_frames: int, jpeg_quality: int) -> list[str]:
    """Extract frames with ffmpeg — robust across all codecs including AV1."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "frame_%04d.jpg")
            # ffmpeg -q:v: 2=best, 31=worst; map jpeg_quality (0-100) to that scale
            qv = max(2, min(31, int(31 - (jpeg_quality / 100.0) * 29)))
            subprocess.run(
                ["ffmpeg", "-i", path,
                 "-vf", f"fps={target_fps}",
                 "-frames:v", str(max_frames),
                 "-q:v", str(qv),
                 pattern, "-y"],
                capture_output=True, timeout=120,
            )
            files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))[:max_frames]
            result = []
            for f in files:
                with open(f, "rb") as fh:
                    result.append("data:image/jpeg;base64," + base64.b64encode(fh.read()).decode("ascii"))
            return result
    except Exception:
        return []


def _sample_via_opencv(path: str, target_fps: float, max_frames: int, jpeg_quality: int) -> list[str]:
    """OpenCV fallback — sequential read avoids the seek-triggered hardware-decode path."""
    cap = cv2.VideoCapture(path)
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        if src_fps > 0 and total > 0:
            step = max(1, int(round(src_fps / max(target_fps, 1e-6))))
            indices = set(range(0, total, step)[:max_frames])
            idx = 0
            ok, frame = cap.read()
            while ok:
                if idx in indices:
                    frames.append(frame)
                    if len(frames) >= max_frames:
                        break
                idx += 1
                ok, frame = cap.read()
        else:
            ok, frame = cap.read()
            while ok and len(frames) < max_frames:
                frames.append(frame)
                ok, frame = cap.read()
        result = []
        for frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                result.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"))
        return result
    finally:
        cap.release()
