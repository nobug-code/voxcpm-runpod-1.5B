import base64
import io
import os
import requests

import numpy as np
import soundfile as sf
import torch
import torchaudio
import runpod
from voxcpm import VoxCPM
from transformers.trainer_utils import set_seed

# --- Env ---
VOXCPM_MODEL = os.getenv("VOXCPM_MODEL", "openbmb/VoxCPM1.5")
ZIPENHANCER_PATH = os.getenv("ZIPENHANCER_PATH", "/runpod-volume/zipenhancer")
ENABLE_DENOISER = os.getenv("ENABLE_DENOISER", "1") == "1"
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en")

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load once at startup ---
print(f"[VoxCPM] Loading model '{VOXCPM_MODEL}' on {device}...")
model = VoxCPM.from_pretrained(
    hf_model_id=VOXCPM_MODEL,
    load_denoiser=ENABLE_DENOISER,
    zipenhancer_model_id=ZIPENHANCER_PATH,
)

# Store sample rate from the model
SAMPLE_RATE = model.tts_model.sample_rate


def split_text_chunks(text: str, max_length: int = 1400) -> list:
    """
    Split text into chunks that don't exceed the maximum character limit.
    Tries to split at sentence boundaries when possible.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining_text = text

    while remaining_text:
        if len(remaining_text) <= max_length:
            chunks.append(remaining_text)
            break

        # Try to find a good breaking point (sentence end)
        break_point = max_length
        for i in range(max_length - 1, max(0, max_length - 200), -1):
            if i < len(remaining_text) and remaining_text[i] in '.!?':
                if i == len(remaining_text) - 1 or (i + 1 < len(remaining_text) and remaining_text[i + 1] == ' '):
                    break_point = i + 1
                    break

        # If no good sentence break found, try to break at a space
        if break_point == max_length:
            for i in range(max_length - 1, max(0, max_length - 100), -1):
                if remaining_text[i] == ' ':
                    break_point = i
                    break

        chunk = remaining_text[:break_point].strip()
        if chunk:
            chunks.append(chunk)

        remaining_text = remaining_text[break_point:].lstrip()

    return chunks


def synthesize_speech(text: str, prompt_text: str = None, prompt_wav_path: str = None, language: str = None, cfg_value_input: float = 2.0, inference_timesteps: int = 10, max_tokenlength: int = 4096) -> str:
    """
    Generate speech audio from text using VoxCPM and return base64-encoded WAV.
    Handles text splitting for inputs exceeding 1400 characters.
    """
    text_chunks = split_text_chunks(text, max_length=1400)

    all_audio_chunks = []

    for i, chunk in enumerate(text_chunks):
        print(f"[VoxCPM] Processing chunk {i+1}/{len(text_chunks)} ({len(chunk)} characters)")

        current_prompt_text = prompt_text

        chunk_audio_chunks = []
        for chunk_audio in model.generate_streaming(chunk, prompt_wav_path, current_prompt_text, cfg_value_input, inference_timesteps, max_tokenlength):
            chunk_audio_chunks.append(chunk_audio)

        if not chunk_audio_chunks:
            raise RuntimeError(f"VoxCPM did not return any audio chunks for chunk {i+1}.")

        all_audio_chunks.extend(chunk_audio_chunks)

    if not all_audio_chunks:
        raise RuntimeError("VoxCPM did not return any audio chunks.")

    audio = np.concatenate(all_audio_chunks)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def download_wav(url: str, save_path: str):
    """Downloads a WAV file from a URL and saves it to a specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded WAV from {url} to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading WAV from {url}: {e}")
        raise


def handler(job):

    job_input = job.get("input", {})
    text = job_input.get("text")
    prompt_text = job_input.get('prompt_text', None)
    prompt_wav_url = job_input.get('prompt_wav_url', None)
    inference_timesteps = job_input.get('inference_timesteps', 10)
    cfg_value_input = job_input.get('cfg_value_input', 2.0)
    max_tokenlength = job_input.get('max_tokenlength', 4096)
    prompt_wav_path = None

    if prompt_wav_url:
        custom_wav_folder = "/workspace/customwav"
        os.makedirs(custom_wav_folder, exist_ok=True)
        filename = os.path.basename(prompt_wav_url)
        if not filename:
            filename = "downloaded_prompt.wav"
        prompt_wav_path = os.path.join(custom_wav_folder, filename)
        download_wav(prompt_wav_url, prompt_wav_path)

    language = job_input.get("language", DEFAULT_LANGUAGE)

    if not isinstance(text, str) or not text.strip():
        return {"error": "Missing required 'text' (non-empty string)."}

    try:
        audio_b64 = synthesize_speech(text.strip(), prompt_text, prompt_wav_path, language, cfg_value_input, inference_timesteps, max_tokenlength)
        return {"language": language, "audio_base64": audio_b64}
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")


# Check if the script is run directly (for local testing or execution)
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

# This line is crucial for RunPod serverless execution
runpod.serverless.start({"handler": handler})
