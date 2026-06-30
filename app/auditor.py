"""Phase-2 gray-zone auditor: Qwen-VL judges clips the frontrunner couldn't auto-decide.

Calls Qwen-VL through Alibaba Cloud Model Studio's OpenAI-compatible endpoint
(QWEN_MODEL_CLOUD_API_URL / QWEN_MODEL_CLOUD_API_KEY) instead of loading the
model locally — no GPU/torch needed for this phase.

Video frames are sampled client-side and sent as base64 JPEGs inside a
"video" content block. The compatible-mode API's "video_url" type only
accepts a hosted URL, not base64/local paths, so for an uploaded file with no
public URL, a sampled frame sequence is the portable option (and is exactly
the format the API documents for "image list as video").
"""

import json
import re
import threading
from typing import Optional

from openai import OpenAI

from . import config
from .media import has_video_stream, sample_frames_base64

_client_lock = threading.Lock()
_client: Optional[OpenAI] = None

POLICY_DEFAULT = (
    "Policies: Hate/Harassment, Misinformation, Synthetic-Impersonation, "
    "Dangerous-Activities, Graphic-Violence, Sexual-Content, Child-Safety. "
    "Child-Safety is zero-tolerance and overrides everything."
)
SCHEMA_HINT = (
    'Return ONLY minified JSON, no prose, no code fences, with exactly these keys: '
    '{"safety_status":"SAFE|VIOLATION|REVIEW",'
    '"triggered_policy":"<one policy or null>",'
    '"confidence_score":0.0,'
    '"policy_aware_caption":"<=40 words, why",'
    '"violation_timestamp":"MM:SS-MM:SS or null",'
    '"frontrunner_reconciliation":"confirms|overrides|<=20 words on the gap"}'
)


def is_ready() -> bool:
    return bool(config.QWEN_MODEL_CLOUD_API_URL and config.QWEN_MODEL_CLOUD_API_KEY)


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            if not is_ready():
                raise RuntimeError(
                    "QWEN_MODEL_CLOUD_API_URL / QWEN_MODEL_CLOUD_API_KEY are not configured "
                    "(set them in backend/.env.local)"
                )
            _client = OpenAI(
                base_url=config.QWEN_MODEL_CLOUD_API_URL,
                api_key=config.QWEN_MODEL_CLOUD_API_KEY,
                timeout=config.QWEN_REQUEST_TIMEOUT_SECONDS,
            )
    return _client


def _frontrunner_context(fr: Optional[dict]) -> str:
    if not fr:
        return ""
    m = fr["metrics"]
    return (f"\nPhase-1 signals to reconcile: visual={m['visual']:.2f}, text={m['text']:.2f}, "
            f"audio={m['audio']:.2f}; acoustic_match='{fr.get('acoustic_match')}'; "
            f"transcript='{fr.get('extracted_transcript', '')[:300]}'. "
            "If audio flags a phonetic bypass that the visual context exonerates, say so.")


def _parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {"safety_status": "REVIEW", "_raw": raw}
    except json.JSONDecodeError:
        return {"safety_status": "REVIEW", "_parse_error": True, "_raw": raw}


def _generate(messages: list, max_new_tokens: int) -> str:
    client = _get_client()
    completion = client.chat.completions.create(
        model=config.QWEN_MODEL_NAME,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return completion.choices[0].message.content or ""


def deep_audit(
    video_path: str,
    frontrunner: Optional[dict] = None,
    policy: str = POLICY_DEFAULT,
    fps: float = config.QWEN_VIDEO_FPS,
    max_frames: int = config.QWEN_MAX_FRAMES,
    max_new_tokens: int = config.QWEN_MAX_NEW_TOKENS,
) -> dict:
    ctx = _frontrunner_context(frontrunner)
    if has_video_stream(video_path):
        frames = sample_frames_base64(
            video_path, target_fps=fps, max_frames=max_frames, jpeg_quality=config.QWEN_JPEG_QUALITY
        )
        prompt = (f"You are the final arbiter in a content-moderation pipeline. {policy}\n"
                  f"Watch the clip (sampled frames below, in order), then judge it.{ctx}\n{SCHEMA_HINT}")
        content = [
            {"type": "video", "video": frames, "fps": fps},
            {"type": "text", "text": prompt},
        ]
    else:
        # audio-only: no frames to sample, judge from transcript + acoustic match.
        prompt = (f"You are the final arbiter in a content-moderation pipeline. {policy}\n"
                  f"There are no video frames. Judge from the transcript and acoustic signals only.{ctx}\n{SCHEMA_HINT}")
        content = [{"type": "text", "text": prompt}]

    raw = _generate([{"role": "user", "content": content}], max_new_tokens)
    return _parse(raw)
