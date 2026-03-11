FROM runpod/pytorch:2.3.1-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    git git-lfs build-essential ffmpeg sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Clone CosyVoice with submodules
RUN git clone --recurse-submodules https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice

WORKDIR /app/CosyVoice

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/CosyVoice/handler.py
COPY .runpod /app/CosyVoice/.runpod

ENV HF_HOME=/workspace/hf
ENV MODELSCOPE_CACHE=/workspace/modelscope
ENV TOKENIZERS_PARALLELISM=false
ENV COSYVOICE_MODEL=pretrained_models/Fun-CosyVoice3-0.5B-finetune

CMD ["python3", "-u", "handler.py"]
