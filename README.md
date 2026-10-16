An agentic RAG chatbot over my CV, publications, and code repositories.
Ask questions about my background, skills, projects, or research in
natural language and get grounded, sourced answers.

## Stack

- **Streamlit** — chat UI
- **llama-cpp-python** + **Qwen2.5** — local LLM (no API calls)
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **ChromaDB** — persistent vector store
- **PyPDF** + custom section-aware chunking — ingestion

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# On Apple Silicon, install llama-cpp-python with Metal support:
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall

streamlit run app.py
