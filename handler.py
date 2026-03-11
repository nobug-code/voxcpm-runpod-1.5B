import runpod
import sys, os, tempfile, base64
import torch, torchaudio

def load_model():
    sys.path.append('third_party/Matcha-TTS')
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    from cosyvoice.cli.cosyvoice import AutoModel
    return AutoModel(
        model_dir='pretrained_models/Fun-CosyVoice3-0.5B-finetune',
        load_trt=True, load_vllm=True, fp16=False
    )

cosyvoice = load_model()

def handler(job):
    data = job["input"]

    wav_bytes = base64.b64decode(data['prompt_wav_b64'])
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(wav_bytes)
        prompt_wav_path = tmp.name

    try:
        chunks = [
            output['tts_speech']
            for output in cosyvoice.inference_instruct2(
                data['tts_text'],
                data.get('instruct_text', 'You are a helpful assistant.<|endofprompt|>'),
                prompt_wav_path,
                stream=False
            )
        ]
    finally:
        os.unlink(prompt_wav_path)

    audio = torch.cat(chunks, dim=1)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        out_path = tmp.name
    torchaudio.save(out_path, audio, cosyvoice.sample_rate)

    with open(out_path, 'rb') as f:
        result_b64 = base64.b64encode(f.read()).decode()
    os.unlink(out_path)

    return {'wav_b64': result_b64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})