import os
from pathlib import Path

from dotenv import load_dotenv

# backend/.env.local — explicit filename, so it won't be picked up by dotenv's
# default auto-discovery; load it before reading any os.environ.get() below.
# Values already present in the real environment take precedence (override=False).
_ENV_LOCAL = Path(__file__).resolve().parent.parent / ".env.local"
if _ENV_LOCAL.exists():
    load_dotenv(_ENV_LOCAL, override=False)


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---- routing / scoring knobs (same defaults as the source notebook) ----
PASS_POLICY = os.environ.get("PASS_POLICY", "vlm_confirms")  # "vlm_confirms" | "funnel_pass"
HARD_BLOCK_AT = float(os.environ.get("HARD_BLOCK_AT", 0.96))  # raw peak >= this -> fast-kill, skip Qwen

ROUTING = os.environ.get("ROUTING", "discrepancy_gated")  # used only when PASS_POLICY == "funnel_pass"
ALPHA = float(os.environ.get("ALPHA", 0.70))               # max vs RMS blend in r_total
GAMMA = float(os.environ.get("GAMMA", 1.5))                # variance amplifier (reported in s_a)
PASS_AT = float(os.environ.get("PASS_AT", 0.30))            # r_total below this + low discrepancy -> AUTO-PASS
BLOCK_AT = float(os.environ.get("BLOCK_AT", 0.85))          # r_total above this + low discrepancy -> AUTO-BLOCK
DISC_GATE = float(os.environ.get("DISC_GATE", 0.05))        # cross-modal variance above this -> GRAY_ZONE
LEGACY_PASS = float(os.environ.get("LEGACY_PASS", 0.25))    # legacy gate: s_a < this -> AUTO-PASS
LEGACY_BLOCK = float(os.environ.get("LEGACY_BLOCK", 0.85))  # legacy gate: s_a >= this -> AUTO-BLOCK

# ---- frontrunner detector config ----
FRONTRUNNER_DEVICE = os.environ.get("FRONTRUNNER_DEVICE", "cuda")
DANGER_CLASSES = {"knife", "scissors", "baseball bat"}  # COCO danger-ish subset
BANNED_LEXICON = ["hate", "kill", "bomb", "scam"]
DANGER_PHRASES = [
    "someone aggressively shouting the f-word slur",
    "phonetic evasion of a curse word, screaming f u",
    "loud explosive blast sound effect or gunshots",
    "vocal breakdown of someone crying and screaming in distress",
]

# ---- calibration (opt-in, fail-safe — see TrackCalibrator in frontrunner.py) ----
CALIBRATION_METHOD = os.environ.get("CALIBRATION_METHOD", "robust_minmax")  # robust_minmax | ecdf | zlogistic
MIN_CALIB_SAMPLES = int(os.environ.get("MIN_CALIB_SAMPLES", 20))  # refuse to fit below this many clips

# ---- Qwen-VL gray-zone auditor (Alibaba Cloud Model Studio, OpenAI-compatible endpoint) ----
QWEN_MODEL_CLOUD_API_URL = os.environ.get("QWEN_MODEL_CLOUD_API_URL", "")
QWEN_MODEL_CLOUD_API_KEY = os.environ.get("QWEN_MODEL_CLOUD_API_KEY", "")
# Must match a model id served by your Model Studio endpoint (verified via
# client.models.list() against QWEN_MODEL_CLOUD_API_URL — "qwen3-vl-plus" is
# DashScope's flagship video-capable Qwen3-VL model; "qwen3-vl-flash" is the
# cheaper/faster sibling if cost matters more than accuracy).
QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "qwen3-vl-plus")
QWEN_VIDEO_FPS = float(os.environ.get("QWEN_VIDEO_FPS", 1.0))       # frame-sampling rate for the "video" content block
QWEN_MAX_FRAMES = int(os.environ.get("QWEN_MAX_FRAMES", 16))        # cap frames/request (token budget + cost)
QWEN_JPEG_QUALITY = int(os.environ.get("QWEN_JPEG_QUALITY", 80))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", 384))
QWEN_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("QWEN_REQUEST_TIMEOUT_SECONDS", 120))

# ---- model lifecycle ----
LOAD_FRONTRUNNER_ON_STARTUP = _flag("LOAD_FRONTRUNNER_ON_STARTUP", True)

# ---- upload / URL-ingestion handling ----
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 500 * 1024 * 1024))  # 500 MB
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4a", ".wav", ".mp3"}
ALLOWED_DOWNLOAD_SCHEMES = {"http", "https"}
DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", 60))

# ---- CORS (frontend origin(s), comma separated; "*" for dev) ----
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]
