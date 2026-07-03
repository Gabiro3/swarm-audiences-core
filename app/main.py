"""Swarm Audiences moderation API.

frontrunner (cheap filters) -> router -> Qwen-VL gray-zone auditor -> unified verdict.

Endpoints:
  GET  /health               readiness of the frontrunner stack / Qwen-VL client / calibrator
  POST /v1/triage            submit a phase-1 triage job; returns {job_id} immediately
  POST /v1/audit             submit a phase-2 Qwen-VL audit job; returns {job_id} immediately
  POST /v1/moderate          submit a full pipeline job; returns {job_id} immediately
  GET  /v1/jobs/{job_id}     poll a job — status: pending | running | done | error
  POST /v1/calibrate         fit the per-track score calibrator on a batch of background clips

Every moderation endpoint accepts the clip as EITHER a multipart file upload
("file") OR a hosted URL ("video_url" form field) — not both.
"""

import json
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import auditor, config, frontrunner, ingest, jobs, orchestrator, schemas

logger = logging.getLogger("swarm_audiences")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.LOAD_FRONTRUNNER_ON_STARTUP:
        frontrunner.load_frontrunner()
    if not auditor.is_ready():
        logger.warning(
            "QWEN_MODEL_CLOUD_API_URL / QWEN_MODEL_CLOUD_API_KEY not set — "
            "/v1/audit and /v1/moderate will fail until backend/.env.local is configured."
        )
    yield


app = FastAPI(
    title="Swarm Audiences Moderation API",
    description=(
        "Triage + deep-audit video moderation pipeline: cheap multi-modal "
        "frontrunner filters route gray-zone clips to a Qwen-VL auditor, "
        "producing one unified PUBLISH/BLOCK/REVIEW verdict as JSON."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in config.ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix}'")

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="swarm_upload_")
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file too large")
                out.write(chunk)
    except HTTPException:
        os.remove(path)
        raise
    return path


def _resolve_media(file: Optional[UploadFile], video_url: Optional[str]) -> str:
    if file and video_url:
        raise HTTPException(400, "provide either 'file' or 'video_url', not both")
    if file:
        return _save_upload(file)
    if video_url:
        return ingest.download_video(video_url)
    raise HTTPException(400, "provide either 'file' (multipart upload) or 'video_url'")


def _cleanup(*paths: str) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _run_job(job_id: str, fn, *args) -> None:
    """Background thread entry point: set_running → fn(*args) → set_done/error."""
    jobs.set_running(job_id)
    try:
        result = fn(*args)
        # Pydantic models → dict so they're JSON-serialisable in the job store
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        jobs.set_done(job_id, result)
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        jobs.set_error(job_id, str(exc))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=schemas.HealthStatus)
def health():
    return schemas.HealthStatus(
        status="ok",
        frontrunner_ready=frontrunner.is_ready(),
        qwen_ready=auditor.is_ready(),
        calibration_fitted=frontrunner.CALIBRATOR.fitted,
    )


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------

@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    """Poll a previously submitted moderation job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found or has expired (TTL={jobs.JOB_TTL}s)")
    return job


# ---------------------------------------------------------------------------
# Moderation endpoints — all return {job_id, status:"pending"} immediately
# ---------------------------------------------------------------------------

@app.post("/v1/triage")
def triage_endpoint(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
):
    """Submit a phase-1 triage job. Poll GET /v1/jobs/{job_id} for the result."""
    path = _resolve_media(file, video_url)
    job_id = jobs.create("triage")

    def run():
        try:
            _run_job(job_id, frontrunner.frontrunner_triage, path)
        finally:
            _cleanup(path)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "status": "pending"}


@app.post("/v1/audit")
def audit_endpoint(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    frontrunner_context: Optional[str] = Form(
        None, description="Optional JSON-encoded frontrunner triage result to reconcile against."
    ),
):
    """Submit a phase-2 Qwen-VL audit job. Poll GET /v1/jobs/{job_id} for the result."""
    fr_ctx = None
    if frontrunner_context:
        try:
            fr_ctx = json.loads(frontrunner_context)
        except json.JSONDecodeError:
            raise HTTPException(400, "frontrunner_context must be valid JSON")

    path = _resolve_media(file, video_url)
    job_id = jobs.create("audit")

    def run():
        try:
            _run_job(job_id, auditor.deep_audit, path, fr_ctx)
        finally:
            _cleanup(path)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "status": "pending"}


@app.post("/v1/moderate")
def moderate_endpoint(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
):
    """Submit a full pipeline job (triage + escalation). Poll GET /v1/jobs/{job_id} for the result."""
    path = _resolve_media(file, video_url)
    job_id = jobs.create("moderate")

    def run():
        try:
            _run_job(job_id, orchestrator.moderate, path)
        finally:
            _cleanup(path)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "status": "pending"}


# ---------------------------------------------------------------------------
# Calibration (kept synchronous — admin/batch operation, not user-facing)
# ---------------------------------------------------------------------------

@app.post("/v1/calibrate", response_model=schemas.CalibrationReport)
def calibrate_endpoint(files: List[UploadFile] = File(...)):
    """Fit the per-track score calibrator on a batch of background clips."""
    paths = [_save_upload(f) for f in files]
    try:
        calibrator, rows, errors = frontrunner.fit_calibration(paths)
        return schemas.CalibrationReport(
            method=calibrator.method,
            fitted=calibrator.fitted,
            fitted_on=len(rows),
            skipped=errors,
            bounds=(
                {t: {"p_low": lo, "p_high": hi} for t, (lo, hi) in calibrator.lohi.items()}
                if calibrator.fitted else {}
            ),
        )
    finally:
        _cleanup(*paths)
