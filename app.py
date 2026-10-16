import streamlit as st
from pathlib import Path

from src.config import DATA_DIR, COLLECTION_NAME, MODEL_FILE, EMBED_MODEL_NAME
from src.llm import load_llm
from src.embeddings import load_embedder
from src.vectorstore import get_chroma_client, get_collection, build_collection, retrieve
from src.ingest import load_documents, chunk_documents
from src.agent import agentic_decide, generate_response

st.set_page_config(page_title="Career RAG", page_icon="💼", layout="wide")
st.title("Chat with My Career Portfolio")
st.caption("Ask about my GitHub projects, publications, or CV. Local LLM + Agentic RAG.")



# --- Sidebar ---
with st.sidebar:
    st.header("Knowledge Base")
    files = [f for f in Path(DATA_DIR).rglob("*") if f.is_file()]
    st.write(f"**{len(files)}** file(s) in `data/`")
    with st.expander("Show files"):
        for f in files:
            st.write(f"- `{f.relative_to(DATA_DIR)}`")

    if st.button("Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Loading documents..."):
            docs = load_documents()
        if not docs:
            st.error("No documents found in `data/`.")
        else:
            with st.spinner("Embedding and indexing..."):
                embedder = load_embedder()
                chunks = chunk_documents(docs)
                texts = [c.page_content for c in chunks]
                metas = [c.metadata for c in chunks]
                embs = embedder.encode(texts, show_progress_bar=False)
                build_collection(texts, embs, metas)
            st.success(f"Indexed {len(docs)} docs → {len(texts)} chunks.")

    st.markdown("---")
    st.caption(f"**Model:** `{MODEL_FILE}`\n\n**Embeddings:** `{EMBED_MODEL_NAME}`")
    if st.button("️Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()



# --- Load resources ---
with st.spinner("Loading model (first run downloads ~2 GB, cached after)..."):
    llm = load_llm()
embedder = load_embedder()
collection = get_collection()



if collection is None:
    st.info("No knowledge base yet. Click **Rebuild Knowledge Base** in the sidebar.")



# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])



if prompt := st.chat_input("Ask about my projects, publications, or CV..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            use_kb = agentic_decide(llm, prompt) and collection is not None
            chunks, metas = [], []
            if use_kb:
                res = retrieve(prompt, embedder, collection,
                               focus_high_signal=True)
                chunks = res["documents"][0] if res["documents"] else []
                metas  = res["metadatas"][0] if res["metadatas"] else []
            answer = generate_response(llm, prompt, chunks, context_meta=metas)
            st.markdown(answer)

            if use_kb and chunks:
                with st.expander(f"📚 Sources ({len(chunks)} chunks)"):
                    for i, (c, m) in enumerate(zip(chunks, metas), 1):
                        prov = m.get("provenance") or Path(m.get("source", "?")).name
                        section = m.get("section", "?")
                        st.markdown(f"**Chunk {i}** — `{prov}`  ·  section: `{section}`")
                        st.caption(c[:600] + ("..." if len(c) > 600 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
