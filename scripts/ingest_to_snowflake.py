import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from snowflake.snowpark import Session
from src.ingest import load_documents, chunk_documents
from src.config import CHUNKS_TABLE, EMBED_MODEL

from snowflake.snowpark import Session
from src.ingest import load_documents, chunk_documents
from src.config import CHUNKS_TABLE, EMBED_MODEL

session = Session.builder.config("connection_name", "connections").create()

chunks = chunk_documents(load_documents())
rows = [
    (f"chunk_{i}",
     c.metadata["source"], c.metadata["provenance"],
     c.metadata["section"], c.metadata["section_type"], c.page_content)
    for i, c in enumerate(chunks)
]

df = session.create_dataframe(
    rows,
    schema=["chunk_id", "source", "provenance",
            "section", "section_type", "content"],
)
df.write.save_as_table(CHUNKS_TABLE, mode="overwrite")

session.sql(f"""
    UPDATE {CHUNKS_TABLE}
    SET embedding = SNOWFLAKE.CORTEX.AI_EMBED('{EMBED_MODEL}', content)
    WHERE embedding IS NULL
""").collect()

print(f"Loaded {len(rows)} chunks into {CHUNKS_TABLE}")