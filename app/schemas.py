from typing import Optional

from pydantic import BaseModel, ConfigDict


class TrackMetrics(BaseModel):
    visual: float
    text: float
    audio: float


class AnnotatedFrame(BaseModel):
    timestamp_s: float
    image: str  # data:image/jpeg;base64,... — boxes/labels drawn via `supervision`


class TriageResult(BaseModel):
    verdict: str
    adjusted_risk_score: float
    r_total: float
    discrepancy: float
    metrics: TrackMetrics
    metrics_calibrated: Optional[TrackMetrics] = None
    routed_on: str
    acoustic_match: str
    extracted_transcript: str
    has_audio: bool
    visual_detections: list = []
    all_detected_objects: list = []
    annotated_frames: list[AnnotatedFrame] = []
    text_flagged_segments: list = []


class AuditResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    safety_status: str
    triggered_policy: Optional[str] = None
    confidence_score: Optional[float] = None
    policy_aware_caption: Optional[str] = None
    violation_timestamp: Optional[str] = None
    frontrunner_reconciliation: Optional[str] = None


class ModerationResult(BaseModel):
    final: str
    decided_by: str
    frontrunner: TriageResult
    audit: Optional[AuditResult] = None


class CalibrationBounds(BaseModel):
    p_low: float
    p_high: float


class CalibrationReport(BaseModel):
    method: str
    fitted: bool
    fitted_on: int
    skipped: list
    bounds: dict[str, CalibrationBounds]


class HealthStatus(BaseModel):
    status: str
    frontrunner_ready: bool
    qwen_ready: bool
    calibration_fitted: bool
