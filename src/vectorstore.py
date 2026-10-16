
import streamlit as st
import chromadb

from .config import CHROMA_PATH, COLLECTION_NAME, TOP_K


EXCLUDE_SECTIONS = [
    "introduction", "background",
    "references", "acknowledgments", "appendix",
]


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection():
    client = get_chroma_client()
    try:
        return client.get_collection(COLLECTION_NAME)
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
             focus_high_signal=True):
    """
    Retrieve top-k chunks for `query`.

    focus_high_signal=True skips Introduction, Background, References,
    Acknowledgments, Appendix, and front matter. These are almost never
    useful for "what can Karl do?" style questions and they dilute the
    context window.

    If the filtered query returns nothing, we retry without the section
    filter — so an unusually-phrased query can still find relevant text.
    """
    q_emb = embedder.encode([query]).tolist()

    if focus_high_signal:
        where = {
            "$and": [
                {"section_type": "own"},
                {"section": {"$nin": EXCLUDE_SECTIONS}},
            ]
        }
    else:
        where = {"section_type": "own"}

    def _query(w):
        return collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            where=w,
            include=["documents", "metadatas", "distances"],
        )

    try:
        res = _query(where)
    except Exception as e:
        # Chroma occasionally rejects complex filters on older builds.
        print(f"[vectorstore] filtered query failed: {e}")
        res = _query({"section_type": "own"})

    # If the focus filter returned nothing (shouldn't happen, but be safe),
    # retry without it so the user still gets *some* context.
    if focus_high_signal and (not res["documents"] or not res["documents"][0]):
        res = _query({"section_type": "own"})

    return res