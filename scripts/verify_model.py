"""Smoke test — download, load, and run one inference. Run before
first `streamlit run app.py` to catch model/HF issues in isolation.

Usage:  python scripts/verify_model.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.config import MODEL_REPO, MODEL_FILE, N_CTX, N_GPU_LAYERS


def main():
    print(f"[1/3] Downloading {MODEL_FILE}")
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"       → {path}")

    print(f"[2/3] Loading model (n_gpu_layers={N_GPU_LAYERS})")
    llm = Llama(
        model_path=path,
        n_ctx=N_CTX,
        n_threads=os.cpu_count(),
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    print("       loaded")

    print("[3/3] Inferring")
    out = llm("Q: What is 2+2?\nA:", max_tokens=8, temperature=0.0, echo=False)
    print(f"       → {out['choices'][0]['text'].strip()!r}")
    print("\n✅ All good. Try `streamlit run app.py`.")


if __name__ == "__main__":
    main()
