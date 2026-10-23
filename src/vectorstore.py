"""ChromaDB client with section-filtered retrieval and entity-lock."""
import re
import streamlit as st
import chromadb

from .config import CHROMA_PATH, COLLECTION_NAME, TOP_K, EXCLUDE_SECTIONS


_ENTITY_PATTERNS = [
    re.compile(r"\bat\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})"),
    re.compile(r"\b(?:the|about|on|in)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+project\b"),
    re.compile(r"\babout\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b"),
]

_STOPWORDS = {
    "what", "does", "have", "with", "karl", "list", "tell", "about",
    "from", "this", "that", "there", "project", "projects", "skills",
    "skill", "work", "did", "any", "the", "for", "and", "use",
}


def _extract_entities(query: str) -> list[str]:
    out = []
    for pat in _ENTITY_PATTERNS:
        for m in pat.findall(query):
            m = m.strip()
            if m and m.lower() not in _STOPWORDS:
                out.append(m)
    for tok in re.findall(r"\b([a-z]{4,15})\b", query):
        if tok not in _STOPWORDS:
            out.append(tok)
    return list(dict.fromkeys(out))


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection():
    try:
        return get_chroma_client().get_collection(COLLECTION_NAME)
    except Exception:
        return None


def build_collection(texts, embeddings, metadatas):
    client = get_chroma_client()
    col = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
    )
    existing = col.get()
    if existing["ids"]:
        col.delete(ids=existing["ids"])
    col.add(
        ids=[f"chunk_{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    return col


def retrieve(query, embedder, collection, top_k=TOP_K,
             focus_high_signal=True, entity_lock=True):
    q_emb = embedder.encode([query]).tolist()

    if focus_high_signal:
        where = {"$and": [
            {"section_type": "own"},
            {"section": {"$nin": list(EXCLUDE_SECTIONS)}},
        ]}
    else:
        where = {"section_type": "own"}

    # --- Semantic pass ---------------------------------------------------
    def _query(n, doc_filter=None):
        kwargs = dict(
            query_embeddings=q_emb,
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        if doc_filter:
            kwargs["where_document"] = doc_filter
        return collection.query(**kwargs)

    try:
        sem = _query(top_k)
    except Exception as e:
        print(f"[vectorstore] semantic query failed: {e}")
        sem = _query(top_k)  # retry without filter

    docs = list(sem["documents"][0]) if sem["documents"] else []
    metas = list(sem["metadatas"][0]) if sem["metadatas"] else []

    # --- Keyword pass (fixes OCR / project-name queries) -----------------
    if entity_lock:
        ents = _extract_entities(query)
        seen = {d[:120].lower() for d in docs}

        for ent in ents:
            if len(ent) < 3:
                continue
            try:
                kw = _query(top_k, doc_filter={"$contains": ent})
            except Exception:
                # Some Chroma versions require exact case; try variations
                try:
                    kw = _query(top_k, doc_filter={"$contains": ent.capitalize()})
                except Exception:
                    continue

            for d, m in zip(kw["documents"][0], kw["metadatas"][0]):
                key = d[:120].lower()
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
                    metas.append(m)

    # Deduplicate and trim
    final_docs, final_metas, seen = [], [], set()
    for d, m in zip(docs, metas):
        key = d[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(d)
        final_metas.append(m)
        if len(final_docs) >= top_k:
            break

    return {"documents": [final_docs], "metadatas": [final_metas]}