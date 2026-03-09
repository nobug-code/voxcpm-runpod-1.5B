"""
RunPod Serverless Handler for VoxCPM1.5 TTS

Supports:
  - Text-to-Speech (TTS): text → audio
  - Voice Cloning: text + prompt_wav (base64) + prompt_text → audio

Input schema:
{
  "input": {
    "text": "string (required)",
    "prompt_wav": "base64-encoded WAV string (optional, for voice cloning)",
    "prompt_text": "string (optional, required if prompt_wav is provided)",
    "cfg_value": float (default 2.0),
    "inference_timesteps": int (default 10),
    "normalize": bool (default false),
    "denoise": bool (default false),
    "retry_badcase": bool (default true),
    "output_format": "wav" | "mp3" (default "wav")
  }
}

Output:
{
  "audio_base64": "base64-encoded audio string",
  "sample_rate": int,
  "duration_seconds": float,
  "output_format": "wav" | "mp3"
}

Environment variables:
  MODEL_PATH: path to VoxCPM1.5 model dir (default: /runpod-volume/VoxCPM1.5)
  ZIPENHANCER_PATH: path to ZipEnhancer model (default: /runpod-volume/zipenhancer)
  ENABLE_DENOISER: "1" to enable denoiser (default: "1")
"""

import os
import io
import base64
import tempfile
import traceback

import numpy as np
import soundfile as sf
import runpod

# ──────────────────────────────────────────────
# Model paths from env (Network Volume mounts)
# ──────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/VoxCPM1.5")
ZIPENHANCER_PATH = os.environ.get("ZIPENHANCER_PATH", "/runpod-volume/zipenhancer")
ENABLE_DENOISER = os.environ.get("ENABLE_DENOISER", "1") == "1"

# ──────────────────────────────────────────────
# Lazy global model (loaded once on cold start)
# ──────────────────────────────────────────────
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    from voxcpm import VoxCPM

    print(f"[VoxCPM] Loading model from: {MODEL_PATH}")
    print(f"[VoxCPM] ZipEnhancer path: {ZIPENHANCER_PATH}")
    print(f"[VoxCPM] Denoiser enabled: {ENABLE_DENOISER}")

    zipenhancer = ZIPENHANCER_PATH if ENABLE_DENOISER else None

    _model = VoxCPM(
        voxcpm_model_path=MODEL_PATH,
        zipenhancer_model_path=zipenhancer,
        enable_denoiser=ENABLE_DENOISER,
        optimize=True,
    )
    print("[VoxCPM] Model loaded and warmed up.")
    return _model


def wav_array_to_bytes(wav: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "mp3":
        # soundfile doesn't support mp3 write; use pydub if available
        try:
            from pydub import AudioSegment
            pcm_buf = io.BytesIO()
            sf.write(pcm_buf, wav, sample_rate, format="WAV", subtype="PCM_16")
            pcm_buf.seek(0)
            seg = AudioSegment.from_wav(pcm_buf)
            seg.export(buf, format="mp3")
        except ImportError:
            # fallback to wav if pydub is not installed
            sf.write(buf, wav, sample_rate, format="WAV")
            fmt = "wav"
    else:
        sf.write(buf, wav, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read(), fmt


def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    # ── required ──────────────────────────────
    text = job_input.get("text", "").strip()
    if not text:
        return {"error": "input.text is required and must be non-empty"}

    # ── optional voice-cloning fields ─────────
    prompt_wav_b64: str | None = job_input.get("prompt_wav")
    prompt_text: str | None = job_input.get("prompt_text")

    if bool(prompt_wav_b64) != bool(prompt_text):
        return {"error": "prompt_wav and prompt_text must both be provided or both be omitted"}

    # ── generation params ─────────────────────
    cfg_value = float(job_input.get("cfg_value", 2.0))
    inference_timesteps = int(job_input.get("inference_timesteps", 10))
    normalize = bool(job_input.get("normalize", False))
    denoise = bool(job_input.get("denoise", False))
    retry_badcase = bool(job_input.get("retry_badcase", True))
    output_format = job_input.get("output_format", "wav").lower()
    if output_format not in ("wav", "mp3"):
        output_format = "wav"

    try:
        model = load_model()
    except Exception as e:
        return {"error": f"Model loading failed: {e}", "traceback": traceback.format_exc()}

    # ── write prompt wav to tmp file if provided ──
    tmp_prompt_path = None
    try:
        if prompt_wav_b64:
            wav_bytes = base64.b64decode(prompt_wav_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_prompt_path = f.name

        wav: np.ndarray = model.generate(
            text=text,
            prompt_wav_path=tmp_prompt_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
        )

        sample_rate: int = model.tts_model.sample_rate
        duration = len(wav) / sample_rate

        audio_bytes, output_format = wav_array_to_bytes(wav, sample_rate, output_format)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 3),
            "output_format": output_format,
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

    finally:
        if tmp_prompt_path and os.path.exists(tmp_prompt_path):
            os.unlink(tmp_prompt_path)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load model before RunPod starts accepting jobs (reduces cold start latency)
    load_model()
    runpod.serverless.start({"handler": handler})
