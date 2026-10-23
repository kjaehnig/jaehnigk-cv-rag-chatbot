from pathlib import Path

# --- Author ---
AUTHOR_NAME = "Karl Jaehnigk"
AUTHOR_CITATION_PATTERNS = [r"Jaehnigk", r"J\.\s*Jaehnigk"]

# --- Model ---
# 1.5B fits comfortably in memory and downloads in ~2 min on a fresh run.
MODEL_REPO = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "career_docs"

# --- RAG params ---
CHUNK_SIZE = 1000
TOP_K = 5
MAX_NEW_TOKENS = 512
N_CTX = 8192
N_GPU_LAYERS = -1   # -1 offloads to Metal on Apple Silicon; 0 for CPU

SUPPORTED_EXTS = {
    ".pdf", ".txt", ".md", ".py", ".ipynb",
    ".json", ".rst", ".html", ".tex",
}

# Sections excluded from retrieval by default. These don't describe
# what Karl can actually do.
EXCLUDE_SECTIONS = (
    "introduction", "background",
    "references", "acknowledgments", "appendix",
)