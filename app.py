"""Career RAG — local Streamlit app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import AUTHOR_NAME
from src.llm import load_llm, make_completer
from src.embeddings import load_embedder
from src.vectorstore import get_collection, build_collection, retrieve
from src.ingest import load_documents, chunk_documents

from src.agent import agentic_decide, generate_response,_wants_list
from src.config import (
    AUTHOR_NAME, MODEL_FILE, EMBED_MODEL_NAME, DATA_DIR, TOP_K,
)

from pathlib import Path


"""Streamlit chat interface — shared by app.py."""




def render_chat(retrieve_fn, complete_fn) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input(f"Ask about {AUTHOR_NAME}'s work..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                use_kb = agentic_decide(complete_fn, prompt)
                chunks, metas = [], []
                if use_kb:
                    res = retrieve_fn(prompt, top_k=TOP_K)
                    chunks = res["documents"][0] if res["documents"] else []
                    metas  = res["metadatas"][0] if res["metadatas"] else []

                answer = generate_response(complete_fn, prompt, chunks, metas)

                # Retry if the model truncated a list request to < 3 items
                if _wants_list(prompt) and answer.count("-") < 2 and len(answer.split("\n")) < 3:
                    if len(chunks) > 0:
                        stricter = (
                            prompt
                            + "\n\nIMPORTANT: The context above contains multiple skills. "
                              "List EVERY skill mentioned. Minimum 5 bullets. One per line, "
                              "each starting with '- '."
                        )
                        stricter_answer = generate_response(complete_fn, stricter, chunks, metas)
                        if stricter_answer.count("-") >= answer.count("-"):
                            answer = stricter_answer
                st.markdown(answer)

                if use_kb and chunks:
                    with st.expander(f"📚 Sources ({len(chunks)} chunks)"):
                        for i, (c, m) in enumerate(zip(chunks, metas), 1):
                            prov = m.get("provenance", "?")
                            sec  = m.get("section", "?")
                            st.markdown(f"**Chunk {i}** — `{prov}` · `{sec}`")
                            st.caption(c[:600] + ("..." if len(c) > 600 else ""))

        st.session_state.messages.append({"role": "assistant", "content": answer})


def render_sidebar(rebuild_fn) -> None:
    with st.sidebar:
        st.header("Knowledge Base")

        files = [f for f in Path(DATA_DIR).rglob("*") if f.is_file()]
        st.write(f"**{len(files)}** file(s) in `data/`")
        with st.expander("Show files"):
            for f in files:
                st.write(f"- `{f.relative_to(DATA_DIR)}`")

        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            rebuild_fn()

        st.markdown("---")
        st.caption(f"**Model:** `{MODEL_FILE}`\n\n**Embeddings:** `{EMBED_MODEL_NAME}`")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

st.set_page_config(page_title="Career RAG", page_icon="💼", layout="wide")
st.title(f"Chat with {AUTHOR_NAME}'s Portfolio")
st.caption("Ask about projects, publications, or CV. Local LLM + Agentic RAG.")


def rebuild_kb():
    with st.spinner("Loading documents..."):
        docs = load_documents()
    if not docs:
        st.error("No documents found in `data/`.")
        return
    with st.spinner("Embedding and indexing..."):
        embedder = load_embedder()
        chunks = chunk_documents(docs)
        texts = [c.page_content for c in chunks]
        metas = [c.metadata for c in chunks]
        embs = embedder.encode(texts, show_progress_bar=False)
        build_collection(texts, embs, metas)
    st.success(f"Indexed {len(docs)} sections → {len(texts)} chunks.")


# --- Sidebar with rebuild button ---
render_sidebar(rebuild_kb)

# --- Load resources ---
with st.spinner("Loading model (first run downloads ~2 GB, cached after)..."):
    llm = load_llm()
complete_fn = make_completer(llm)

embedder = load_embedder()
collection = get_collection()

if collection is None:
    st.info("👈 No knowledge base yet. Click **Rebuild Knowledge Base** in the sidebar.")

# --- Adapter: bind the local components into the shape ui.py expects ---
def retrieve_fn(query: str, top_k: int = 5) -> dict:
    return retrieve(query, embedder, collection, top_k=top_k,
                    focus_high_signal=True, entity_lock=True)


# --- Chat loop ---
render_chat(retrieve_fn=retrieve_fn, complete_fn=complete_fn)