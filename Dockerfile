# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# ffmpeg      - frontrunner shells out to it directly for audio extraction.
# libgl1 /
# libglib2.0-0 - required by opencv-python-headless at runtime despite the
#                "headless" build (still links against these).
# libsndfile1 - librosa's audio decode backend.
# curl        - used by the HEALTHCHECK below.
# gosu        - drop from root to appuser in docker-entrypoint.sh, after
#               fixing ownership of whatever got mounted at container start.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        curl \
        gosu \
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
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# strip any CRLF line endings (Windows editors/git autocrlf) before chmod —
# a shebang line ending in \r fails silently as "no such file or directory"
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R appuser:appuser /app

# Don't COPY .env.local here (see .dockerignore) — pass QWEN_MODEL_CLOUD_API_URL /
# QWEN_MODEL_CLOUD_API_KEY / QWEN_MODEL_NAME as real container env vars or a
# secret store at run time instead of baking credentials into the image.

# Stay root here — docker-entrypoint.sh fixes ownership of whatever Railway
# (or `docker run -v`) mounts at /home/appuser/.cache, then drops to appuser
# before exec'ing CMD. A plain `USER appuser` would lose write access the
# moment a volume gets mounted over that directory at container start.

EXPOSE 8000

# Model weights (~hundreds of MB) land in $HOME/.cache on first request, and
# get re-downloaded on every fresh container otherwise. No Docker VOLUME
# instruction here — attach a Railway Volume mounted at /home/appuser/.cache
# (Settings -> Volumes on this service) for that to persist across deploys.

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
