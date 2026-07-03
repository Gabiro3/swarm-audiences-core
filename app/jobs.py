"""In-memory job store for async moderation requests.

POST endpoints return a job_id immediately; the actual ML work runs in a
background daemon thread. Callers poll GET /v1/jobs/{job_id} until status
is "done" or "error". Jobs expire after JOB_TTL seconds to avoid unbounded
memory growth.
"""

import threading
import time
import uuid
from typing import Optional

JOB_TTL = 1800  # 30 minutes

_lock = threading.Lock()
_store: dict[str, dict] = {}


def create(job_type: str) -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _store[job_id] = {
            "job_id": job_id,
            "type": job_type,
            "status": "pending",
            "created_at": time.time(),
            "result": None,
            "error": None,
        }
    return job_id


def set_running(job_id: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "running"


def set_done(job_id: str, result: dict) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "done"
            _store[job_id]["result"] = result


def set_error(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "error"
            _store[job_id]["error"] = error


def get(job_id: str) -> Optional[dict]:
    with _lock:
        job = _store.get(job_id)
        if job is None:
            return None
        if time.time() - job["created_at"] > JOB_TTL:
            del _store[job_id]
            return None
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
            "elapsed_s": round(time.time() - job["created_at"], 1),
        }
