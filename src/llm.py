import os
import streamlit as st
from huggingface_hub import hf_hub_download
from .config import MODEL_REPO, MODEL_FILE, N_CTX


@st.cache_resource(show_spinner=False)
def _download_model() -> str:
    return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)


@st.cache_resource(show_spinner=False)
def load_llm():
    from llama_cpp import Llama
    path = _download_model()
    return Llama(
        model_path=path,
        n_ctx=N_CTX,
        n_threads=os.cpu_count(),
        n_gpu_layers=-1,   # Metal on Mac, ignored on CPU-only hosts
        verbose=False,
    )
