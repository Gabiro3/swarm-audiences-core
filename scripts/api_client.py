"""Thin client for the Swarm Audiences moderation API, used by benchmark_runner.py.

Submits a clip (local file upload or hosted URL) to POST /v1/moderate, then
polls GET /v1/jobs/{job_id} until it reaches a terminal status. One
`requests.Session` is created per worker thread (see benchmark_runner.py) so
connections are pooled without sharing a Session across threads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests


class ModerationAPIError(Exception):
    pass


@dataclass
class ModerationOutcome:
    job_id: Optional[str]
    status: str  # "done" | "error" | "timeout"
    result: Optional[dict]
    error: Optional[str]
    client_latency_s: float
    server_elapsed_s: Optional[float]


def submit_and_wait(
    session: requests.Session,
    api_base: str,
    *,
    file_path: Optional[str] = None,
    video_url: Optional[str] = None,
    endpoint: str = "/v1/moderate",
    poll_interval_s: float = 2.0,
    submit_timeout_s: float = 120.0,
    overall_timeout_s: float = 900.0,
) -> ModerationOutcome:
    """Submit one clip and block until its job finishes (or times out).

    Exactly one of file_path / video_url must be given, mirroring the API's
    own "file" xor "video_url" contract.
    """
    if bool(file_path) == bool(video_url):
        raise ValueError("submit_and_wait: pass exactly one of file_path or video_url")

    t0 = time.monotonic()
    url = api_base.rstrip("/") + endpoint

    try:
        if video_url:
            resp = session.post(url, data={"video_url": video_url}, timeout=submit_timeout_s)
        else:
            with open(file_path, "rb") as fh:
                files = {"file": (file_path.rsplit("/", 1)[-1], fh)}
                resp = session.post(url, files=files, timeout=submit_timeout_s)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except Exception as exc:  # noqa: BLE001 - surfaced as a row, not a crash
        return ModerationOutcome(
            job_id=None, status="error", result=None,
            error=f"submit failed: {type(exc).__name__}: {exc}",
            client_latency_s=time.monotonic() - t0, server_elapsed_s=None,
        )

    poll_url = f"{api_base.rstrip('/')}/v1/jobs/{job_id}"
    while True:
        if time.monotonic() - t0 > overall_timeout_s:
            return ModerationOutcome(
                job_id=job_id, status="timeout", result=None,
                error=f"exceeded overall_timeout_s={overall_timeout_s}",
                client_latency_s=time.monotonic() - t0, server_elapsed_s=None,
            )
        try:
            poll_resp = session.get(poll_url, timeout=30)
            poll_resp.raise_for_status()
            job = poll_resp.json()
        except Exception as exc:  # noqa: BLE001 - transient network hiccup, keep polling
            time.sleep(poll_interval_s)
            continue

        if job["status"] == "done":
            return ModerationOutcome(
                job_id=job_id, status="done", result=job["result"], error=None,
                client_latency_s=time.monotonic() - t0, server_elapsed_s=job.get("elapsed_s"),
            )
        if job["status"] == "error":
            return ModerationOutcome(
                job_id=job_id, status="error", result=None, error=job.get("error", "unknown error"),
                client_latency_s=time.monotonic() - t0, server_elapsed_s=job.get("elapsed_s"),
            )
        time.sleep(poll_interval_s)
