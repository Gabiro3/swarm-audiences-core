import threading

# Frontrunner (YOLO/Whisper/CLAP) and the Qwen3-VL auditor typically share one GPU.
# Serialize all model inference behind this lock so concurrent API requests don't
# race on CUDA memory/context.
inference_lock = threading.Lock()
