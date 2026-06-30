"""Swarm Audiences moderation API.

frontrunner (cheap filters) -> router -> Qwen-VL gray-zone auditor -> unified verdict.

Endpoints:
  GET  /health         readiness of the frontrunner stack / Qwen-VL client / calibrator
  POST /v1/triage       phase-1 only: cheap multi-modal triage for a video/audio clip
  POST /v1/audit        phase-2 only: Qwen-VL deep audit (optionally given frontrunner context)
  POST /v1/moderate      full pipeline: triage, escalate to deep audit if needed, final verdict
  POST /v1/calibrate     fit the per-track score calibrator on a batch of background clips

Every moderation endpoint accepts the clip as EITHER a multipart file upload
("file") OR a hosted URL ("video_url" form field) — not both.
"""

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import auditor, config, frontrunner, ingest, orchestrator, schemas

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
            os.remove(path)


@app.get("/health", response_model=schemas.HealthStatus)
def health():
    return schemas.HealthStatus(
        status="ok",
        frontrunner_ready=frontrunner.is_ready(),
        qwen_ready=auditor.is_ready(),
        calibration_fitted=frontrunner.CALIBRATOR.fitted,
    )


@app.post("/v1/triage", response_model=schemas.TriageResult)
def triage_endpoint(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
):
    """Phase-1 triage only: cheap visual/text/audio filters, no VLM call."""
    path = _resolve_media(file, video_url)
    try:
        return frontrunner.frontrunner_triage(path)
    finally:
        _cleanup(path)


@app.post("/v1/audit", response_model=schemas.AuditResult)
def audit_endpoint(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    frontrunner_context: Optional[str] = Form(
        None, description="Optional JSON-encoded frontrunner triage result to reconcile against."
    ),
):
    """Phase-2 deep audit only: runs Qwen-VL directly on the clip."""
    fr_ctx = None
    if frontrunner_context:
        try:
            fr_ctx = json.loads(frontrunner_context)
        except json.JSONDecodeError:
            raise HTTPException(400, "frontrunner_context must be valid JSON")

    path = _resolve_media(file, video_url)
    try:
        return auditor.deep_audit(path, fr_ctx)
    finally:
        _cleanup(path)


@app.post("/v1/moderate", response_model=schemas.ModerationResult)
def moderate_endpoint(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
):
    """Full pipeline: triage, escalate to deep audit only if gray-zone, unified verdict."""
    path = _resolve_media(file, video_url)
    try:
        return orchestrator.moderate(path)
    finally:
        _cleanup(path)


@app.post("/v1/calibrate", response_model=schemas.CalibrationReport)
def calibrate_endpoint(files: List[UploadFile] = File(...)):
    """Fit the per-track score calibrator on a batch of background clips.

    Pass a held-out set of known-benign clips — never the clips you're about
    to judge. Below MIN_CALIB_SAMPLES the fit is refused and routing stays on
    raw (uncalibrated) scores, which is the safe fallback.
    """
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
