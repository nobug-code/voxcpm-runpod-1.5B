# voxcpm-runpod

[![Runpod](https://api.runpod.io/badge/nobug-code/voxcpm-runpod-1.5B)](https://console.runpod.io/hub/nobug-code/voxcpm-runpod-1.5B)

RunPod Serverless worker for [VoxCPM1.5](https://github.com/OpenBMB/VoxCPM) — tokenizer-free TTS with voice cloning.

## Features

- **TTS**: text → WAV/MP3
- **Voice Cloning**: text + reference audio → WAV/MP3
- Model loaded once per worker (low latency after cold start)
- Model weights served from RunPod **Network Volume** (no re-download on restart)

---

## 1. Prepare Network Volume

Create a Network Volume in RunPod and attach it. Then download the model weights onto the volume (run once from a GPU pod that has the volume attached):

```bash
# Install deps
pip install huggingface_hub modelscope

# VoxCPM1.5 weights  →  /runpod-volume/VoxCPM1.5
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("openbmb/VoxCPM1.5", local_dir="/runpod-volume/VoxCPM1.5")
EOF

# ZipEnhancer weights  →  /runpod-volume/zipenhancer
python - <<'EOF'
from modelscope import snapshot_download
snapshot_download("iic/speech_zipenhancer_ans_multiloss_16k_base",
                  local_dir="/runpod-volume/zipenhancer")
EOF
```

---

## 2. Build & Push Docker Image

```bash
docker build -t your-dockerhub-user/voxcpm-runpod:latest .
docker push your-dockerhub-user/voxcpm-runpod:latest
```

---

## 3. Create RunPod Serverless Endpoint

1. Go to **RunPod → Serverless → + New Endpoint**
2. Set **Container Image** to `your-dockerhub-user/voxcpm-runpod:latest`
3. Under **Environment Variables**, set:
   | Key | Value |
   |-----|-------|
   | `MODEL_PATH` | `/runpod-volume/VoxCPM1.5` |
   | `ZIPENHANCER_PATH` | `/runpod-volume/zipenhancer` |
   | `ENABLE_DENOISER` | `1` |
4. Under **Volume**, attach your Network Volume and set mount path to `/runpod-volume`
5. Choose GPU (RTX 4090 recommended for RTF ~0.15)

---

## 4. API Usage

### TTS (text only)

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello, this is VoxCPM speaking.",
      "cfg_value": 2.0,
      "inference_timesteps": 10,
      "output_format": "wav"
    }
  }'
```

### Voice Cloning

Encode your reference WAV as base64 first:

```bash
PROMPT_B64=$(base64 -w 0 reference.wav)

curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"text\": \"Hello, cloning your voice!\",
      \"prompt_wav\": \"$PROMPT_B64\",
      \"prompt_text\": \"Text spoken in the reference audio.\",
      \"cfg_value\": 2.0,
      \"inference_timesteps\": 10,
      \"output_format\": \"wav\"
    }
  }"
```

### Decode response audio (Python)

```python
import base64, requests

resp = requests.post(
    "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync",
    headers={"Authorization": "Bearer <RUNPOD_API_KEY>"},
    json={"input": {"text": "Hello from VoxCPM!"}}
).json()

audio_bytes = base64.b64decode(resp["output"]["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

---

## Input Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | **required** | Text to synthesize |
| `prompt_wav` | string (base64) | `null` | Reference WAV for voice cloning |
| `prompt_text` | string | `null` | Transcript of reference audio |
| `cfg_value` | float | `2.0` | Guidance scale (higher = closer to prompt) |
| `inference_timesteps` | int | `10` | Diffusion steps (higher = better quality) |
| `normalize` | bool | `false` | Enable text normalization |
| `denoise` | bool | `false` | Denoise prompt audio before cloning |
| `retry_badcase` | bool | `true` | Auto-retry on bad generation |
| `output_format` | `"wav"` \| `"mp3"` | `"wav"` | Output audio format |

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `audio_base64` | string | Base64-encoded audio |
| `sample_rate` | int | Audio sample rate (44100 for VoxCPM1.5) |
| `duration_seconds` | float | Duration of generated audio |
| `output_format` | string | Actual format returned |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/runpod-volume/VoxCPM1.5` | Path to VoxCPM1.5 model on Network Volume |
| `ZIPENHANCER_PATH` | `/runpod-volume/zipenhancer` | Path to ZipEnhancer model |
| `ENABLE_DENOISER` | `1` | Set to `0` to disable denoiser |
