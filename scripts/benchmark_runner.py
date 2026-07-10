#!/usr/bin/env python3
"""Benchmark runner: streams clips from one or more public datasets through the
Swarm Audiences moderation API and appends one CSV row per clip.

    Datasets
        |
        v
    Python Benchmark Script (this file)
        |
        +-- OpenVid / WebVid / XD-Violence / UCF-Crime / HD-VILA
        v
    Moderation API (POST /v1/moderate, GET /v1/jobs/{id})
        |
        v
    results.csv (appended, never overwritten)

Each row: timestamp, video_id, source, ground_truth, prediction, confidence,
latency_s, reasoning, decided_by, job_id, status, error, server_elapsed_s.

See README.md in this directory for full usage, or run --help.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import queue
import sys
import threading
import time
from typing import Iterator, Optional

import requests

from api_client import submit_and_wait
from dataset_adapters import (
    ClipRef,
    iter_hdvila,
    iter_labeled_dir,
    iter_manifest_csv,
    iter_openvid,
    iter_openvid_hf,
    iter_ucf_crime_kaggle,
    iter_webvid,
    iter_webvid_hf,
    iter_xd_violence,
)

CSV_FIELDS = [
    "timestamp", "video_id", "source", "ground_truth", "prediction", "confidence",
    "latency_s", "reasoning", "decided_by", "job_id", "status", "error", "server_elapsed_s",
]

_csv_lock = threading.Lock()
_progress_lock = threading.Lock()
_counts = {"done": 0, "error": 0, "timeout": 0}


def _extract_row(clip: ClipRef, outcome) -> dict:
    prediction = confidence = reasoning = decided_by = ""
    if outcome.status == "done" and outcome.result:
        result = outcome.result
        prediction = result.get("final", "")
        decided_by = result.get("decided_by", "")
        audit = result.get("audit")
        frontrunner = result.get("frontrunner") or {}
        if audit:
            confidence = audit.get("confidence_score", "")
            reasoning = audit.get("policy_aware_caption", "") or audit.get("frontrunner_reconciliation", "")
        else:
            confidence = frontrunner.get("adjusted_risk_score", "")
            reasoning = (
                f"phase1 auto-decision (routed_on={frontrunner.get('routed_on')}, "
                f"r_total={frontrunner.get('r_total')}, acoustic_match={frontrunner.get('acoustic_match')})"
            )
    else:
        prediction = outcome.status.upper()  # ERROR / TIMEOUT
        reasoning = outcome.error or ""

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_id": clip.video_id,
        "source": clip.source,
        "ground_truth": clip.ground_truth,
        "prediction": prediction,
        "confidence": confidence,
        "latency_s": round(outcome.client_latency_s, 3),
        "reasoning": reasoning,
        "decided_by": decided_by,
        "job_id": outcome.job_id or "",
        "status": outcome.status,
        "error": outcome.error or "",
        "server_elapsed_s": outcome.server_elapsed_s if outcome.server_elapsed_s is not None else "",
    }


def _append_row(csv_path: str, row: dict) -> None:
    with _csv_lock:
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            fh.flush()


def _worker(work_q: "queue.Queue[Optional[ClipRef]]", args: argparse.Namespace) -> None:
    session = requests.Session()  # one connection pool per worker thread
    while True:
        clip = work_q.get()
        if clip is None:
            work_q.task_done()
            return
        outcome = submit_and_wait(
            session, args.api_base,
            file_path=clip.file_path, video_url=clip.video_url,
            endpoint=args.endpoint,
            poll_interval_s=args.poll_interval,
            submit_timeout_s=args.submit_timeout,
            overall_timeout_s=args.overall_timeout,
        )
        _append_row(args.output, _extract_row(clip, outcome))
        with _progress_lock:
            _counts[outcome.status] = _counts.get(outcome.status, 0) + 1
            total = sum(_counts.values())
            if total % 10 == 0 or outcome.status != "done":
                print(
                    f"[{total}] {clip.source}/{clip.video_id} -> {outcome.status} "
                    f"({outcome.client_latency_s:.1f}s)  done={_counts['done']} "
                    f"error={_counts.get('error', 0)} timeout={_counts.get('timeout', 0)}",
                    file=sys.stderr,
                )
        work_q.task_done()


def _build_dataset_iterators(args: argparse.Namespace) -> list[Iterator[ClipRef]]:
    iterators = []
    for name in args.datasets:
        if name == "webvid":
            it = iter_webvid(args.webvid_csv)
        elif name == "webvid_hf":
            it = iter_webvid_hf(split=args.webvid_hf_split)
        elif name == "openvid":
            it = iter_openvid(args.openvid_csv, args.openvid_video_dir)
        elif name == "openvid_hf":
            it = iter_openvid_hf(split=args.openvid_hf_split)
        elif name == "ucf_crime":
            it = iter_labeled_dir(args.ucf_crime_dir, source="ucf_crime")
        elif name == "ucf_crime_kaggle":
            it = iter_ucf_crime_kaggle(dataset_slug=args.ucf_crime_kaggle_slug)
        elif name == "xd_violence":
            it = iter_xd_violence(args.xd_violence_dir)
        elif name == "hdvila":
            it = iter_hdvila(
                args.hdvila_jsonl, args.hdvila_clips_dir, fetch_missing=args.hdvila_fetch
            )
        elif name == "manifest":
            it = iter_manifest_csv(args.manifest_csv, source=args.manifest_source)
        else:
            raise ValueError(f"unknown dataset {name!r}")
        if args.limit is not None:
            it = itertools.islice(it, args.limit)
        iterators.append(it)
    return iterators


def _shard(clips: Iterator[ClipRef], shard_index: int, shard_count: int) -> Iterator[ClipRef]:
    if shard_count <= 1:
        yield from clips
        return
    for i, clip in enumerate(clips):
        if i % shard_count == shard_index:
            yield clip


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark the moderation API against public video datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-base", default="http://localhost:8000", help="Moderation API base URL")
    p.add_argument("--endpoint", default="/v1/moderate", choices=["/v1/moderate", "/v1/triage", "/v1/audit"])
    p.add_argument(
        "--datasets", nargs="+", required=True,
        choices=[
            "webvid", "webvid_hf",
            "openvid", "openvid_hf",
            "ucf_crime", "ucf_crime_kaggle",
            "xd_violence", "hdvila", "manifest",
        ],
        help="One or more dataset adapters to pull clips from",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap PER DATASET (not overall) — omit for all clips")
    p.add_argument("--workers", type=int, default=4, help="Concurrent worker threads submitting+polling jobs")
    p.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between job-status polls")
    p.add_argument("--submit-timeout", type=float, default=120.0, help="Timeout for the initial POST (upload)")
    p.add_argument("--overall-timeout", type=float, default=900.0, help="Give up on a clip after this many seconds")
    p.add_argument("--output", default="results.csv", help="CSV path — always appended, never overwritten")
    p.add_argument("--shard-index", type=int, default=0, help="This process's shard, for multi-process/-machine runs")
    p.add_argument("--shard-count", type=int, default=1, help="Total number of shards (see README for parallel runs)")

    p.add_argument("--webvid-csv", help="WebVid results CSV (has a contentUrl column) — local file")
    p.add_argument("--webvid-hf-split", default="train", help="HF split for webvid_hf adapter (train or validation)")
    p.add_argument("--openvid-csv", help="OpenVid-1M manifest CSV (has a video/filename column)")
    p.add_argument("--openvid-video-dir", help="Directory of already-extracted OpenVid-1M .mp4 files")
    p.add_argument("--openvid-hf-split", default="train", help="HF split for openvid_hf adapter")
    p.add_argument("--ucf-crime-kaggle-slug", default="minmints/ufc-crime-full-dataset",
                   help="Kaggle dataset slug for ucf_crime_kaggle adapter")
    p.add_argument("--ucf-crime-dir", help="Root of UCF-Crime's <Category>/<clip>.mp4 layout")
    p.add_argument("--xd-violence-dir", help="Root directory of XD-Violence .mp4 files")
    p.add_argument("--hdvila-jsonl", help="HD-VILA-100M metadata JSONL (video/clip ids + spans)")
    p.add_argument("--hdvila-clips-dir", help="Directory of already-extracted HD-VILA clips (or download target)")
    p.add_argument(
        "--hdvila-fetch", action="store_true",
        help="Shell out to yt-dlp+ffmpeg for HD-VILA clips missing locally (slow; subject to YouTube's ToS)",
    )
    p.add_argument("--manifest-csv", help="Generic CSV with columns video_id, ground_truth, path/url")
    p.add_argument("--manifest-source", default="manifest", help="Source label to tag rows from --manifest-csv")

    args = p.parse_args()

    required = {
        "webvid": ["webvid_csv"], "webvid_hf": [],
        "openvid": ["openvid_csv", "openvid_video_dir"], "openvid_hf": [],
        "ucf_crime": ["ucf_crime_dir"], "ucf_crime_kaggle": [],
        "xd_violence": ["xd_violence_dir"],
        "hdvila": ["hdvila_jsonl", "hdvila_clips_dir"], "manifest": ["manifest_csv"],
    }
    for ds in args.datasets:
        for attr in required[ds]:
            if not getattr(args, attr):
                p.error(f"--datasets {ds} requires --{attr.replace('_', '-')}")

    clips = itertools.chain.from_iterable(_build_dataset_iterators(args))
    clips = _shard(clips, args.shard_index, args.shard_count)

    work_q: "queue.Queue[Optional[ClipRef]]" = queue.Queue(maxsize=args.workers * 4)
    threads = [threading.Thread(target=_worker, args=(work_q, args), daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    t_start = time.monotonic()
    n_queued = 0
    for clip in clips:
        work_q.put(clip)
        n_queued += 1
    for _ in threads:
        work_q.put(None)  # sentinel per worker
    work_q.join()

    elapsed = time.monotonic() - t_start
    print(
        f"\nDone: {n_queued} clips queued in {elapsed:.1f}s "
        f"(done={_counts['done']} error={_counts.get('error', 0)} timeout={_counts.get('timeout', 0)}) "
        f"-> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
