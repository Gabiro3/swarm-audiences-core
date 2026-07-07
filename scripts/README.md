# Benchmark runner

Streams clips from public datasets through the moderation API and appends
one row per clip to a CSV. Built for reproducible benchmarking, not for the
web UI's one-video-at-a-time flow.

```
Datasets --> benchmark_runner.py --> POST /v1/moderate + poll /v1/jobs/{id} --> results.csv (append-only)
```

## Setup

```bash
cd backend/scripts
pip install -r requirements.txt   # just `requests`; the API server has its own deps
python -m app.main                # ...or however you already run the API — it must be reachable at --api-base
```

## CSV schema

`timestamp, video_id, source, ground_truth, prediction, confidence, latency_s, reasoning, decided_by, job_id, status, error, server_elapsed_s`

- `prediction` — final verdict (`PUBLISH`/`BLOCK`/`REVIEW`), or `ERROR`/`TIMEOUT` if the job never completed.
- `confidence` — Qwen's `confidence_score` when phase-2 ran (`decided_by=phase2`), else the frontrunner's `adjusted_risk_score` (`decided_by=phase1`).
- `reasoning` — Qwen's `policy_aware_caption`, or a one-line summary of which phase-1 signal triggered an auto-decision.
- `latency_s` — wall-clock time this client spent from POST to job completion (includes upload + queueing + inference). `server_elapsed_s` is the server's own job-age counter, for comparison.
- The file is **appended to, never overwritten** — safe to re-run, resume, or merge multiple runs into one CSV.

## Per-dataset setup

Each dataset ships in a different shape, so read the caveats in
[`dataset_adapters.py`](dataset_adapters.py) before trusting ground-truth
labels blindly — especially XD-Violence's filename-label parsing, which is a
best-effort community convention, not an official spec.

| Dataset | What you need locally | Flags |
|---|---|---|
| WebVid-10M | Official CSV only (has a direct `contentUrl`) — **no video download needed**, referenced straight by URL | `--datasets webvid --webvid-csv results_10M.csv` |
| OpenVid-1M | Official manifest CSV + the `.mp4`s already extracted from the HF archive parts | `--datasets openvid --openvid-csv OpenVid-1M.csv --openvid-video-dir ./openvid_videos` |
| UCF-Crime | Official release layout `root/<Category>/<clip>.mp4` (folder name = label) | `--datasets ucf_crime --ucf-crime-dir ./UCF_Crimes` |
| XD-Violence | `.mp4`s with the community `..._label_B2-0-0.mp4` naming convention | `--datasets xd_violence --xd-violence-dir ./XD-Violence` |
| HD-VILA-100M | Metadata JSONL (YouTube id + clip span) **and** clips already extracted to `--hdvila-clips-dir/{clip_id}.mp4` — this dataset is YouTube-sourced, not directly downloadable | `--datasets hdvila --hdvila-jsonl hdvila_meta.jsonl --hdvila-clips-dir ./hdvila_clips` |
| Anything else | A CSV you write yourself: `video_id,ground_truth,path` (or `,url`) | `--datasets manifest --manifest-csv my_manifest.csv --manifest-source my_dataset` |

HD-VILA note: if clips aren't extracted yet, add `--hdvila-fetch` to have the
script shell out to `yt-dlp`+`ffmpeg` per missing clip. This is slow and
subject to YouTube's terms of service — prefer pre-extracting clips out of
band for any real benchmark run.

You can mix datasets in one run: `--datasets webvid xd_violence --webvid-csv ... --xd-violence-dir ...`.

## Running it

```bash
python benchmark_runner.py \
  --api-base http://localhost:8000 \
  --datasets webvid xd_violence \
  --webvid-csv ./webvid_results_10M.csv \
  --xd-violence-dir ./XD-Violence \
  --limit 200 \
  --workers 8 \
  --output results.csv
```

`--limit` is applied **per dataset**, not to the combined total (e.g. `--limit 200` with two datasets submits up to 400 clips).

## Parallelism

Two independent knobs:

**1. `--workers` (in-process thread pool).** The work here is almost entirely
I/O-bound (upload, then poll a job that runs on the server/GPU) — Python
threads are the right tool, no multiprocessing needed. `--workers 8` runs 8
clips concurrently. Start around 4-8; raise it if the API server has GPU/queue
headroom, watch for 413/`error` rows piling up if you push too hard.

**2. `--shard-index` / `--shard-count` (multiple processes or machines).** For
higher throughput than one process's thread pool gives you, run several
instances of the script against the *same* dataset args, each with a
different shard index — every clip is deterministically assigned to
`index % shard_count`, so shards never overlap, and all shards can append to
the same `--output results.csv` (row order doesn't matter, and the CSV writer
is lock-protected within a process; give each shard its own output file if
you'd rather merge later with `cat`):

```bash
# 4 parallel processes, e.g. one per terminal tab or one per machine
python benchmark_runner.py --datasets webvid --webvid-csv webvid.csv --shard-count 4 --shard-index 0 --output results_shard0.csv &
python benchmark_runner.py --datasets webvid --webvid-csv webvid.csv --shard-count 4 --shard-index 1 --output results_shard1.csv &
python benchmark_runner.py --datasets webvid --webvid-csv webvid.csv --shard-count 4 --shard-index 2 --output results_shard2.csv &
python benchmark_runner.py --datasets webvid --webvid-csv webvid.csv --shard-count 4 --shard-index 3 --output results_shard3.csv &
wait
cat results_shard*.csv > results.csv   # merge (dedupe the repeated header rows if you care)
```

Each shard can also set its own `--workers`, so total concurrency is
`shard_count * workers`. Scale shards across machines the same way — same
flags, different `--shard-index`, all pointed at the same `--api-base` (or a
load balancer in front of several API replicas).

## Resuming / re-running

The script has no built-in "skip already-processed" logic — it streams the
dataset fresh every run. To resume a large run, use `--limit`/manual slicing,
or `awk -F, '{print $2}' results.csv` (the `video_id` column) to build an
exclude-list and filter it into your manifest before re-running. Kept out of
the script itself to avoid growing a bespoke checkpoint format when a shell
one-liner does the job for a benchmarking tool.
