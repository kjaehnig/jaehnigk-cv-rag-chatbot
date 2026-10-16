from pathlib import Path

AUTHOR_NAME = "Karl JaehnigK"
AUTHOR_CITATION_PATTERNS = [r"Jaehnigk", r"J\.\s*Jaehnigk"]

MODEL_REPO = "bartowski/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "career_docs"

# --- RAG params ---
CHUNK_SIZE = 2000
# CHUNK_OVERLAP = 100
TOP_K = 10
MAX_NEW_TOKENS = 512
N_CTX = 8192

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".py", ".ipynb", ".json", ".rst", ".html", ".tex"}
