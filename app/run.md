From the backend/ directory:


uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
(--port 8000 matches NEXT_PUBLIC_API_BASE_URL in frontend/.env.local.) First-time setup if you haven't already:


pip install -r requirements.txt
Note this installs the full local frontrunner stack (torch, ultralytics, faster-whisper, faiss) which is heavy — and you'll need ffmpeg on PATH for audio extraction, since it isn't installed in this sandbox yet. Once it's up, GET http://localhost:8000/health should show frontrunner_ready: true and qwen_ready: true.