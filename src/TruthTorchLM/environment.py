import os
from pathlib import Path


def get_cache_dir():
    cache_dir = os.getenv("TTLM_HOME", str(
        Path.home() / ".cache" / "TruthTorchLM"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def set_cache_dir(new_dir):
    os.environ["TTLM_HOME"] = new_dir
    os.makedirs(new_dir, exist_ok=True)
    # print(f"Cache directory set to: {new_dir}")
