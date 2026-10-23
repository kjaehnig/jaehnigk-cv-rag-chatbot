"""Local GGUF model loader and completion wrapper."""
import os
import streamlit as st
from huggingface_hub import hf_hub_download

from .config import (
    MODEL_REPO, MODEL_FILE, N_CTX, MAX_NEW_TOKENS, N_GPU_LAYERS,
)


# Stop tokens catch common failure modes before _clean() has to.
_STOP = [
    "\n\n\n\n",
    "\nAnswer:", "\nQuestion:", "\nContext:",
    "\nContext (numbered", "Context (numbered",
    "To determine if", "We would need",
    "The context blocks", "Block [",
    "Would you like me", "Certainly!", "Please provide",
    "How can I help", "Is there anything else",
    "\nHere are", "\nIn summary", "\nIn conclusion",
    "\n```", "```json",
    "Under no circumstances", "If more context",
    # New in this fix:
    "PRONOUNS:", "PROVENANCE:",
    "ROLE-ANCHORED", "Every numbered context block",
    "Always refer to Karl", "Never use he/him",
    "widely recognized", "significant contribution",
    "This work demonstrates", "The project showcases",
    # Old list-phrase stops
    "The user asked for a list",
    "Output a markdown bullet list",
    "every item MUST begin",
    "Do not output bare words",
    "One item per line",
    "Here is the list",
]


@st.cache_resource(show_spinner=False)
def _download_model() -> str:
    return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)


@st.cache_resource(show_spinner=False)
def load_llm():
    from llama_cpp import Llama
    return Llama(
        model_path=_download_model(),
        n_ctx=N_CTX,
        n_threads=os.cpu_count(),
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )


def make_completer(llm):
    """Wrap the raw Llama in a `(prompt) -> str` callable.

    This is what agent.py and ui.py consume, so neither has to know
    anything about llama-cpp-python.
    """
    def _complete(prompt: str) -> str:
        out = llm(
            prompt,
            max_tokens=min(MAX_NEW_TOKENS, 700),
            temperature=0.15,
            top_p=0.9,
            repeat_penalty=1.15,         # was 1.05 — raise for paragraph loops
            presence_penalty=0.1,            
            stop=_STOP,
            echo=False,
        )
        return out["choices"][0]["text"]
    return _complete