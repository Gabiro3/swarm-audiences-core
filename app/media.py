"""Shared video helpers used by both the frontrunner and the Qwen-VL auditor.

Centralized so "does this file have frames" and "sample frames as base64
JPEGs" are implemented once instead of duplicated per module.
"""

import base64

import cv2


def has_video_stream(path: str) -> bool:
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
    """Evenly samples frames at ~target_fps (capped to max_frames) and returns
    them as "data:image/jpeg;base64,..." URIs, newest-to-oldest in time order.

    Used for the Qwen-VL "video" content block: the OpenAI-compatible API only
    accepts video_url as a hosted URL (no base64/local paths), so a sampled
    frame sequence is the portable way to send an uploaded file that has no
    public URL.
    """
    cap = cv2.VideoCapture(path)
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames = []
        if src_fps > 0 and total > 0:
            step = max(1, int(round(src_fps / max(target_fps, 1e-6))))
            indices = list(range(0, total, step))[:max_frames]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        else:
            # metadata missing/unreliable (some containers) — fall back to
            # reading sequentially and taking the first max_frames frames.
            ok, frame = cap.read()
            while ok and len(frames) < max_frames:
                frames.append(frame)
                ok, frame = cap.read()

        data_uris = []
        for frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                data_uris.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"))
        return data_uris
    finally:
        cap.release()
