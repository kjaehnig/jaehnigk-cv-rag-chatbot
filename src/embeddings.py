import streamlit as st
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL_NAME


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)