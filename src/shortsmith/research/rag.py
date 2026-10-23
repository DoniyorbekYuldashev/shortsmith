"""Retrieval: find the few pieces of the knowledge base that fit a topic (the R in RAG).

    data/hooks/*.md --chunk--> sections --embed--> vectors --store--> Chroma (chroma/)
    query ---------------------------embed--> vector --nearest by cosine--> top-k sections

The knowledge base is ~70 hook patterns and 2 brand guidelines (data/hooks/). All of it in
every prompt would cost ~8,000 tokens a call and bury the few patterns that fit; instead each
section is turned into a vector once, and a query brings back only the k closest.

Chunking follows the files' own structure: one "## " section is one chunk (one hook pattern,
or one part of a brand guideline). Every chunk is a whole idea, so a retrieved chunk never
starts or ends mid-thought.

Vectors come from the provider's embedding model through llm.embed(), so every request is
priced and in the ledger. Chroma only stores and searches them: it is never asked to embed
(its default would download a local model). Distance is cosine: the angle between vectors,
not their length, carries the meaning.

Build or refresh the index, then try a query (from the repo root):
    uv run --env-file .env python -m shortsmith.research.rag "cozy cafe in Tashkent" -k 5
"""

import argparse
import hashlib
import re
import statistics
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from shortsmith import llm

if TYPE_CHECKING:  # chromadb takes ~2 s to import: loaded on first use, so the MCP server starts fast
    from chromadb.api.models.Collection import Collection

DATA_DIR = Path("data/hooks")  # relative to the repo root, like runs/
CHROMA_DIR = Path("chroma")  # gitignored: rebuilt from data/hooks/ whenever it's missing
DEFAULT_K = 5

Kind = Literal["hook", "brand"]
Embed = Callable[[list[str]], llm.EmbedResult]

# gemini-embedding-2 has no task_type option. The task goes into the text itself, in the formats
# from Google's embeddings guide: documents get their title, queries are marked as searches.
# Other models (OpenAI's, the test fakes) get the plain text.
TASK_FORMATS: dict[str, tuple[str, str]] = {
    "gemini-embedding-2": ("title: {title} | text: {text}", "task: search result | query: {text}"),
}


class Chunk(BaseModel):
    """One section of the knowledge base: the unit that is embedded, stored and retrieved."""

    id: str  # stable across rebuilds: "curiosity/open-loop", "brands/anor-va-dolchin/voice-and-tone"
    kind: Kind
    title: str  # the section heading: a pattern's name, or "<brand>: <section>"
    group: str  # the file's title: the hook category, or the brand
    brand: str | None  # the brand file's name for brand chunks ("anor-va-dolchin")
    source: str  # file, relative to the data folder
    text: str  # the section, markdown bold removed


class Retrieved(BaseModel):
    id: str
    kind: Kind
    title: str
    group: str
    text: str
    similarity: float  # cosine similarity with the query: 1 = same direction, 0 = unrelated


class IndexReport(BaseModel):
    """What one sync did. Re-running it with nothing changed embeds nothing."""

    collection: str
    chunks: int
    added: int
    changed: int
    removed: int
    unchanged: int
    embedded_tokens: int  # sent to the embedding model by this sync
    cost_usd: float
    seconds: float


# --- chunking -----------------------------------------------------------------------------


def split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """A markdown file's "# " title and its "## " sections as (heading, body) pairs.
    The intro before the first section belongs to no chunk: it describes the whole file."""
    title, sections = "", []
    heading: str | None = None
    body: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        elif line.startswith("# ") and heading is None and not title:
            title = line[2:].strip()
        elif heading is not None:
            body.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(body).strip()))
    return title, sections


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """Every section of every file: hook patterns from data/hooks/*.md, brand sections from
    data/hooks/brands/*.md. Two sections with the same id are an error, not a silent overwrite."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.rglob("*.md")):
        relative = path.relative_to(data_dir)
        is_brand = relative.parts[0] == "brands"
        group, sections = split_sections(path.read_text(encoding="utf-8"))
        for heading, body in sections:
            chunks.append(Chunk(
                id=f"{relative.with_suffix('').as_posix()}/{slug(heading)}",
                kind="brand" if is_brand else "hook",
                title=f"{group}: {heading}" if is_brand else heading,
                group=group,
                brand=path.stem if is_brand else None,
                source=relative.as_posix(),
                text=body.replace("**", ""),
            ))
    ids = [c.id for c in chunks]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"two sections with the same id: {duplicates}")
    return chunks


def document_text(chunk: Chunk, model: str) -> str:
    """The exact string embedded for a chunk. Its hash decides whether a chunk must be re-embedded."""
    doc_format, _ = TASK_FORMATS.get(model, ("{title}\n{text}", "{text}"))
    return doc_format.format(title=chunk.title, text=chunk.text)


def query_text(query: str, model: str) -> str:
    _, query_format = TASK_FORMATS.get(model, ("", "{text}"))
    return query_format.format(text=query)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def collection_name(model: str, dimensions: int) -> str:
    """One collection per embedding model and size: vectors from different models live in
    different spaces and can't be compared, so a provider switch gets its own index."""
    return slug(f"kb-{model}-{dimensions}")


# --- the index ------------------------------------------------------------------------------


class KnowledgeBase:
    """One Chroma collection, kept in step with the files in data/hooks/."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        store: Path | None = CHROMA_DIR,  # None: in memory only (tests)
        model: str | None = None,
        dimensions: int = llm.EMBED_DIMENSIONS,
        embed: Embed | None = None,
    ):
        import chromadb
        from chromadb.config import Settings

        self.data_dir = data_dir
        self.model = model or llm.provider().embedding_model
        self.dimensions = dimensions
        self.embed: Embed = embed or (lambda texts: llm.embed(texts, self.model, dimensions, step="rag"))
        settings = Settings(anonymized_telemetry=False)
        client = (chromadb.PersistentClient(path=str(store), settings=settings) if store
                  else chromadb.EphemeralClient(settings=settings))
        self.collection: Collection = client.get_or_create_collection(
            collection_name(self.model, dimensions),
            embedding_function=None,  # we pass vectors in; Chroma must never embed (or download a model)
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self.model, "dimensions": dimensions},
        )

    def sync(self) -> IndexReport:
        """Make the collection match the files: embed new and changed chunks, delete removed
        ones, leave the rest. Idempotent: a second run with no edits embeds nothing."""
        start = time.perf_counter()
        chunks = load_chunks(self.data_dir)
        texts = {c.id: document_text(c, self.model) for c in chunks}
        stored = self.collection.get(include=["metadatas"])
        stored_hash = {i: (m or {}).get("hash") for i, m in zip(stored["ids"], stored["metadatas"] or [])}

        todo = [c for c in chunks if stored_hash.get(c.id) != fingerprint(texts[c.id])]
        gone = sorted(set(stored_hash) - {c.id for c in chunks})
        tokens, cost = 0, 0.0
        if todo:
            result = self.embed([texts[c.id] for c in todo])
            tokens, cost = result.prompt_tokens, result.cost_usd
            self.collection.upsert(  # insert or replace by id: no duplicates, ever
                ids=[c.id for c in todo],
                embeddings=result.vectors,
                documents=[c.text for c in todo],
                metadatas=[{"kind": c.kind, "title": c.title, "group": c.group, "brand": c.brand or "",
                            "source": c.source, "hash": fingerprint(texts[c.id])} for c in todo],
            )
        if gone:
            self.collection.delete(ids=gone)
        added = sum(c.id not in stored_hash for c in todo)
        return IndexReport(
            collection=self.collection.name, chunks=len(chunks), added=added, changed=len(todo) - added,
            removed=len(gone), unchanged=len(chunks) - len(todo), embedded_tokens=tokens, cost_usd=cost,
            seconds=round(time.perf_counter() - start, 2),
        )

    def brands(self) -> list[str]:
        return sorted({c.brand for c in load_chunks(self.data_dir) if c.brand})

    def retrieve(self, query: str, k: int = DEFAULT_K, kind: Kind | None = "hook", brand: str | None = None) -> list[Retrieved]:
        """The k chunks closest to the query, most similar first. One embedding request."""
        if brand is not None and brand not in (known := self.brands()):
            raise ValueError(f"unknown brand {brand!r}; the brands are: {', '.join(known)}")
        filters = [{"kind": kind}] if kind else []
        if brand:
            filters.append({"brand": brand})
        where = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)
        vector = self.embed([query_text(query, self.model)]).vectors[0]
        found = self.collection.query(query_embeddings=[vector], n_results=k, where=where,
                                      include=["documents", "metadatas", "distances"])
        return [
            Retrieved(id=i, kind=m["kind"], title=m["title"], group=m["group"], text=doc,
                      similarity=round(1 - distance, 4))  # cosine distance = 1 - cosine similarity
            for i, doc, m, distance in zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0])
        ]


@cache
def default_kb() -> KnowledgeBase:
    """The repo's knowledge base with the provider's embedding model, opened once per process."""
    return KnowledgeBase()


def retrieve(query: str, k: int = DEFAULT_K, kind: Kind | None = "hook", brand: str | None = None) -> list[Retrieved]:
    """Retrieve from the repo's knowledge base, syncing it with data/hooks/ first (embeds
    only what changed: nothing, usually, so this costs one query embedding)."""
    kb = default_kb()
    kb.sync()
    return kb.retrieve(query, k, kind, brand)


BRAND_K = 3  # guideline sections, of 5 per brand


def brand_rules(topic: str, brand: str, k: int = BRAND_K) -> list[Retrieved]:
    """A brand's guideline sections for a topic. The query names what a writer needs (tone, words,
    visual style): with the bare topic, "Words we use and avoid" can fall out of the top 3
    (Phase 5 build notes), and those are the rules a script must not break."""
    return retrieve(f"{topic}: tone, words to use and avoid, visual style", k, kind="brand", brand=brand)


def chunk_sizes(chunks: list[Chunk]) -> dict[str, dict[str, int]]:
    """Words and estimated tokens per chunk, by kind: min, median, max, total."""
    sizes: dict[str, dict[str, int]] = {}
    for kind in ("hook", "brand"):
        tokens = [llm.estimate_tokens(c.text) for c in chunks if c.kind == kind]
        words = [len(c.text.split()) for c in chunks if c.kind == kind]
        if tokens:
            sizes[kind] = {"chunks": len(tokens), "min_tokens": min(tokens), "median_tokens": int(statistics.median(tokens)),
                           "max_tokens": max(tokens), "total_tokens": sum(tokens), "median_words": int(statistics.median(words))}
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the hooks knowledge base into Chroma, then retrieve the top k for a query.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--brand", help="search this brand's guideline sections instead of hook patterns")
    args = parser.parse_args()

    kb = default_kb()
    report = kb.sync()
    print(f"index {report.collection}: {report.chunks} chunks | +{report.added} new, {report.changed} changed, "
          f"-{report.removed} removed, {report.unchanged} unchanged | embedded {report.embedded_tokens} tokens "
          f"(${report.cost_usd:.6f} list price) | {report.seconds} s")
    for kind, s in chunk_sizes(load_chunks(kb.data_dir)).items():
        print(f"  {kind}: {s['chunks']} chunks, tokens min {s['min_tokens']} / median {s['median_tokens']} / "
              f"max {s['max_tokens']}, {s['total_tokens']} in total")
    if args.query:
        start = time.perf_counter()
        results = kb.retrieve(args.query, args.k, "brand" if args.brand else "hook", args.brand)
        print(f"\ntop {len(results)} for {args.query!r} ({time.perf_counter() - start:.2f} s):")
        for r in results:
            print(f"  {r.similarity:.4f}  {r.id:<45} {r.title}")


if __name__ == "__main__":
    main()
