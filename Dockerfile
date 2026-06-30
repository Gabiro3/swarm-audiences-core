# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# ffmpeg      - frontrunner shells out to it directly for audio extraction.
# libgl1 /
# libglib2.0-0 - required by opencv-python-headless at runtime despite the
#                "headless" build (still links against these).
# libsndfile1 - librosa's audio decode backend.
# curl        - used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser

# Non-root user. YOLO/Whisper/CLAP weights download into $HOME/.cache on
# first run, so the home directory needs to exist and be writable before
# USER switches to it below.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p "$HOME/.cache" \
    && chown -R appuser:appuser "$HOME"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Don't COPY .env.local here (see .dockerignore) — pass QWEN_MODEL_CLOUD_API_URL /
# QWEN_MODEL_CLOUD_API_KEY / QWEN_MODEL_NAME as real container env vars or a
# secret store at run time instead of baking credentials into the image.

USER appuser

EXPOSE 8000

# Model weights (~hundreds of MB) land in $HOME/.cache on first request, and
# get re-downloaded on every fresh container otherwise. No Docker VOLUME
# instruction here — attach a Railway Volume mounted at /home/appuser/.cache
# (Settings -> Volumes on this service) for that to persist across deploys.

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
