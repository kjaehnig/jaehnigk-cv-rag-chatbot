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

    def _query(w, n=None):
        return collection.query(
            query_embeddings=q_emb,
            n_results=n or top_k,
            where=w,
            include=["documents", "metadatas", "distances"],
        )

    try:
        res = _query(where)
    except Exception as e:
        print(f"[vectorstore] filtered query failed: {e}")
        res = _query({"section_type": "own"})

    if focus_high_signal and (not res["documents"] or not res["documents"][0]):
        res = _query({"section_type": "own"})

    if entity_lock:
        ents = _extract_entities(query)
        if ents:
            big = _query(where, n=max(top_k * 4, 20))
            docs = big["documents"][0] if big["documents"] else []
            metas = big["metadatas"][0] if big["metadatas"] else []

            def score(doc, meta):
                hay = ((meta.get("provenance") or "") + " " + doc).lower()
                return sum(1 for e in ents if e.lower() in hay)

            scored = [(d, m, score(d, m)) for d, m in zip(docs, metas)]
            keep = [(d, m) for d, m, s in scored if s > 0]
            keep.sort(key=lambda dm: -score(dm[0], dm[1]))
            if keep:
                res["documents"][0] = [k[0] for k in keep[:top_k]]
                res["metadatas"][0] = [k[1] for k in keep[:top_k]]

    return res