import os
import io
import base64
import tempfile

import numpy as np
import soundfile as sf
import runpod  # Required

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/VoxCPM1.5")
ZIPENHANCER_PATH = os.environ.get("ZIPENHANCER_PATH", "/runpod-volume/zipenhancer")
ENABLE_DENOISER = os.environ.get("ENABLE_DENOISER", "1") == "1"

# Load model once at worker startup (outside handler to avoid re-initialization)
from voxcpm import VoxCPM

_model = VoxCPM(
    voxcpm_model_path=MODEL_PATH,
    zipenhancer_model_path=ZIPENHANCER_PATH if ENABLE_DENOISER else None,
    enable_denoiser=ENABLE_DENOISER,
    optimize=True,
)


def handler(job):
    job_input = job["input"]

    # Input validation
    text = job_input.get("text", "").strip()
    if not text:
        return {"error": "input.text is required"}

    prompt_wav_b64 = job_input.get("prompt_wav")
    prompt_text = job_input.get("prompt_text")

    if bool(prompt_wav_b64) != bool(prompt_text):
        return {"error": "prompt_wav and prompt_text must both be provided or both be omitted"}

    cfg_value = float(job_input.get("cfg_value", 2.0))
    inference_timesteps = int(job_input.get("inference_timesteps", 10))
    normalize = bool(job_input.get("normalize", False))
    denoise = bool(job_input.get("denoise", False))
    retry_badcase = bool(job_input.get("retry_badcase", True))
    output_format = job_input.get("output_format", "wav").lower()
    if output_format not in ("wav", "mp3"):
        output_format = "wav"

    runpod.serverless.progress_update(job, "Generating speech...")

    tmp_prompt_path = None
    try:
        if prompt_wav_b64:
            wav_bytes = base64.b64decode(prompt_wav_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_prompt_path = f.name

        wav: np.ndarray = _model.generate(
            text=text,
            prompt_wav_path=tmp_prompt_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
        )
    finally:
        if tmp_prompt_path and os.path.exists(tmp_prompt_path):
            os.unlink(tmp_prompt_path)

    runpod.serverless.progress_update(job, "Encoding audio...")

    sample_rate = _model.tts_model.sample_rate
    buf = io.BytesIO()

    if output_format == "mp3":
        try:
            from pydub import AudioSegment
            pcm_buf = io.BytesIO()
            sf.write(pcm_buf, wav, sample_rate, format="WAV", subtype="PCM_16")
            pcm_buf.seek(0)
            AudioSegment.from_wav(pcm_buf).export(buf, format="mp3")
        except ImportError:
            sf.write(buf, wav, sample_rate, format="WAV")
            output_format = "wav"
    else:
        sf.write(buf, wav, sample_rate, format="WAV")

    buf.seek(0)

    return {
        "audio_base64": base64.b64encode(buf.read()).decode("utf-8"),
        "sample_rate": sample_rate,
        "duration_seconds": round(len(wav) / sample_rate, 3),
        "output_format": output_format,
    }


runpod.serverless.start({"handler": handler})  # Required
