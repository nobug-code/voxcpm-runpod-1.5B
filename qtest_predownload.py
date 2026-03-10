import base64
from io import BytesIO

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM
from transformers.trainer_utils import set_seed

model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
text = "Streaming text to speech is easy with VoxCPM!"
sample_rate = model.tts_model.sample_rate

if __name__ == "__main__":
    set_seed(42)
    chunks = []
    for chunk in model.generate_streaming(text):
        chunks.append(chunk)
    final = np.concatenate(chunks)
    buffer = BytesIO()
    sf.write(buffer, final, sample_rate, format="WAV")
    wav_data = buffer.getvalue()

    # Encode to base64
    base64_wav_data = base64.b64encode(wav_data).decode('utf-8')

    # Output the base64 data (e.g., print it)
    print(f"Base64 encoded WAV data: {base64_wav_data}")