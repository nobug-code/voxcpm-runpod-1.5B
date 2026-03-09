import os
import io
import base64
import tempfile
import traceback

import numpy as np
import soundfile as sf
import runpod  # Required

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/VoxCPM1.5")
ZIPENHANCER_PATH = os.environ.get("ZIPENHANCER_PATH", "/runpod-volume/zipenhancer")
ENABLE_DENOISER = os.environ.get("ENABLE_DENOISER", "1") == "1"

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    from voxcpm import VoxCPM

    zipenhancer = ZIPENHANCER_PATH if ENABLE_DENOISER else None
    _model = VoxCPM(
        voxcpm_model_path=MODEL_PATH,
        zipenhancer_model_path=zipenhancer,
        enable_denoiser=ENABLE_DENOISER,
        optimize=True,
    )
    return _model


def handler(event):
    input_data = event["input"]

    text = input_data.get("text", "").strip()
    if not text:
        return {"error": "input.text is required"}

    prompt_wav_b64 = input_data.get("prompt_wav")
    prompt_text = input_data.get("prompt_text")

    if bool(prompt_wav_b64) != bool(prompt_text):
        return {"error": "prompt_wav and prompt_text must both be provided or both be omitted"}

    cfg_value = float(input_data.get("cfg_value", 2.0))
    inference_timesteps = int(input_data.get("inference_timesteps", 10))
    normalize = bool(input_data.get("normalize", False))
    denoise = bool(input_data.get("denoise", False))
    retry_badcase = bool(input_data.get("retry_badcase", True))
    output_format = input_data.get("output_format", "wav").lower()
    if output_format not in ("wav", "mp3"):
        output_format = "wav"

    tmp_prompt_path = None
    try:
        model = load_model()

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

        sample_rate = model.tts_model.sample_rate
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
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "duration_seconds": round(len(wav) / sample_rate, 3),
            "output_format": output_format,
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

    finally:
        if tmp_prompt_path and os.path.exists(tmp_prompt_path):
            os.unlink(tmp_prompt_path)


load_model()
runpod.serverless.start({"handler": handler})  # Required
