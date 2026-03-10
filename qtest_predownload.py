# qtest_predownload.py
import os
from huggingface_hub import snapshot_download

HF_HOME = os.environ.get("HF_HOME", "/workspace/hf")
CACHE_DIR = os.path.join(HF_HOME, "hub")

os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Downloading VoxCPM1.5 to {CACHE_DIR}...")

snapshot_download(
    repo_id="openbmb/VoxCPM1.5",
    cache_dir=CACHE_DIR,
)

print("Download complete.")