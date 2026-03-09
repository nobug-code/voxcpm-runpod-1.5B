# ──────────────────────────────────────────────────────────────
# VoxCPM1.5 — RunPod Serverless Worker
# Base: PyTorch 2.4.0 + CUDA 12.4
# ──────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps (libsndfile for soundfile, ffmpeg for pydub/mp3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# ── Environment defaults (override at RunPod template level) ──
# MODEL_PATH       : where the VoxCPM1.5 weights live on the Network Volume
# ZIPENHANCER_PATH : where ZipEnhancer weights live on the Network Volume
# ENABLE_DENOISER  : "1" to enable, "0" to disable
ENV VOXCPM_MODEL=openbmb/VoxCPM-1.5B


CMD ["python", "-u", "./handler.py"]

