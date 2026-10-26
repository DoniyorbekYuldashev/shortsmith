"""Copy the knowledge base's index from one vector store to another, without embedding anything again.

A vector belongs to the model that made it, not to the database that holds it: the same
gemini-embedding-2 vector means the same thing in Chroma and in Postgres. So moving stores is a
copy of rows (id, metadata, hash, vector), and 0 embedding requests. Because the hash of each
chunk's embedded text travels with it, the next sync() finds every chunk unchanged and embeds 0.

From the repo root (Docker Desktop running, `docker compose up -d db` done):
    uv run --env-file .env python -m shortsmith.research.migrate                 # chroma -> pgvector
    uv run --env-file .env python -m shortsmith.research.migrate --to chroma     # the other way
"""

import argparse
import time

from pydantic import BaseModel

from shortsmith import llm
from shortsmith.research import rag, stores


class CopyReport(BaseModel):
    source: str  # "chroma:kb-gemini-embedding-2-768"
    target: str
    copied: int
    removed: int  # rows in the target that the source doesn't have
    target_rows: int
    hashes_match: bool  # every id with the same hash on both sides
    max_vector_difference: float  # largest |a - b| over every number of every vector: 0.0 = bit-for-bit (float32 both)
    seconds: float


def copy_index(source: stores.VectorStore, target: stores.VectorStore) -> CopyReport:
    """Make `target` hold exactly what `source` holds. Idempotent: a second copy changes nothing."""
    start = time.perf_counter()
    records = source.records()
    target.upsert(records)
    extra = sorted(set(target.hashes()) - {r.id for r in records})
    target.delete(extra)
    copied = {r.id: r for r in target.records()}
    difference = max((abs(a - b) for r in records for a, b in zip(r.embedding, copied[r.id].embedding, strict=True)),
                     default=0.0)
    return CopyReport(
        source=f"{source.backend}:{source.name}", target=f"{target.backend}:{target.name}", copied=len(records),
        removed=len(extra), target_rows=target.count(),
        hashes_match=target.hashes() == {r.id: r.hash for r in records}, max_vector_difference=difference,
        seconds=round(time.perf_counter() - start, 2),
    )


def open_both(model: str, dimensions: int) -> dict[str, stores.VectorStore]:
    """Both stores for one model and size, whatever VECTOR_STORE says (this is the one place that wants both)."""
    collection = rag.collection_name(model, dimensions)
    return {"chroma": stores.ChromaStore(collection, dimensions, model, stores.CHROMA_DIR),
            "pgvector": stores.PgVectorStore(stores.table_name(collection), dimensions)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the RAG index between Chroma and pgvector (0 embedding requests).")
    parser.add_argument("--to", choices=stores.BACKENDS, default="pgvector")
    args = parser.parse_args()

    model, dimensions = llm.embedding_model(), llm.EMBED_DIMENSIONS
    both = open_both(model, dimensions)
    target = both[args.to]
    source = both["chroma" if args.to == "pgvector" else "pgvector"]
    print(f"source {source.backend}:{source.name}: {source.count()} rows | target {target.backend}:{target.name}: "
          f"{target.count()} rows (before)")
    report = copy_index(source, target)
    print(f"copied {report.copied} | removed {report.removed} | target now {report.target_rows} rows | hashes match: "
          f"{report.hashes_match} | max vector difference {report.max_vector_difference:g} | {report.seconds} s")

    def no_embedding(texts: list[str]) -> llm.EmbedResult:
        raise AssertionError(f"sync wanted to embed {len(texts)} texts after the copy")

    sync = rag.KnowledgeBase(store=target, model=model, dimensions=dimensions, embed=no_embedding).sync()
    print(f"sync on the target: +{sync.added} new, {sync.changed} changed, -{sync.removed} removed, "
          f"{sync.unchanged} unchanged | embedded {sync.embedded_tokens} tokens")


if __name__ == "__main__":
    try:
        main()
    except stores.StoreUnavailable as e:  # one line and the next step, not a traceback
        raise SystemExit(f"error: {e}") from None
