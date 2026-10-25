"""Vector stores for the tests: an in-memory fake, Chroma in a temp folder, and (only when the database
is there) Postgres + pgvector in throwaway tables.

The pgvector tests are integration tests (marker `pgvector`). They run when DATABASE_URL is set and the
database answers, e.g. from the repo root with Docker Desktop running and `docker compose up -d db`:
    uv run --env-file .env pytest                 everything, pgvector included
    uv run --env-file .env pytest -m pgvector     only the integration tests
and are skipped, with the reason, otherwise (plain `uv run pytest` has no DATABASE_URL). They never touch
the real index: every table they make is called test_<random>_<name> and is dropped afterwards.
"""

import math
import os
import uuid

import pytest

from shortsmith.research import stores


def cosine_distance(a: list[float], b: list[float]) -> float:
    return 1 - sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


class MemoryStore:
    """A VectorStore in a dict, searched exactly: shows KnowledgeBase needs nothing but the protocol."""

    backend = "memory"

    def __init__(self, name: str, rows: dict[str, stores.Record]):
        self.name = name
        self.rows = rows

    def count(self) -> int:
        return len(self.rows)

    def hashes(self) -> dict[str, str]:
        return {i: r.hash for i, r in self.rows.items()}

    def upsert(self, records: list[stores.Record]) -> None:
        self.rows.update({r.id: r.model_copy(deep=True) for r in records})

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.rows.pop(i, None)

    def query(self, vector: list[float], k: int, kind: str | None, brand: str | None) -> list[stores.Hit]:
        pool = [r for r in self.rows.values() if (not kind or r.kind == kind) and (not brand or r.brand == brand)]
        ranked = sorted(pool, key=lambda r: cosine_distance(vector, r.embedding))[:k]
        return [stores.Hit(id=r.id, kind=r.kind, title=r.title, group=r.group, text=r.text,
                           distance=cosine_distance(vector, r.embedding)) for r in ranked]

    def records(self) -> list[stores.Record]:
        return sorted(self.rows.values(), key=lambda r: r.id)


@pytest.fixture(scope="session")
def pg_url() -> str:
    """DATABASE_URL, if the pgvector database answers; otherwise every test that needs it is skipped."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        pytest.skip("pgvector integration test: DATABASE_URL is not set (start Docker Desktop, "
                    "`docker compose up -d db`, then `uv run --env-file .env pytest`)")
    try:
        stores.connect(url).close()
    except stores.StoreUnavailable as e:
        pytest.skip(f"pgvector integration test: {e.message} (start Docker Desktop, then `docker compose up -d db`)")
    return url


class StoreFactory:
    """Opens stores of one backend by name. Opening a name again gives the same data (a 'restart')."""

    def __init__(self, backend: str, tmp_path, request: pytest.FixtureRequest, dimensions: int):
        self.backend, self.tmp_path, self.request, self.dimensions = backend, tmp_path, request, dimensions
        self.memory: dict[str, dict[str, stores.Record]] = {}
        self.prefix = f"test_{uuid.uuid4().hex[:8]}"
        self.tables: set[str] = set()
        self.opened: list = []

    def open(self, name: str = "kb", dimensions: int | None = None) -> stores.VectorStore:
        dimensions = dimensions or self.dimensions
        if self.backend == "memory":
            return MemoryStore(name, self.memory.setdefault(name, {}))
        if self.backend == "chroma":
            return stores.ChromaStore(f"kb-{name}", dimensions, "fake", self.tmp_path / "chroma")
        url = self.request.getfixturevalue("pg_url")
        table = f"{self.prefix}_{name}"
        self.tables.add(table)
        store = stores.PgVectorStore(table, dimensions, url)
        self.opened.append(store)
        return store

    def cleanup(self) -> None:
        for store in self.opened:
            store.close()
        if self.tables:
            from psycopg import sql

            with stores.connect(self.request.getfixturevalue("pg_url")) as conn:
                for table in self.tables:  # only our own throwaway tables
                    assert table.startswith("test_")
                    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))


@pytest.fixture(params=["memory", "chroma", pytest.param("pgvector", marks=pytest.mark.pgvector)])
def store_factory(request, tmp_path):
    """The same tests against every store: the fake, Chroma, and pgvector (when the database is up)."""
    factory = StoreFactory(request.param, tmp_path, request, dimensions=64)
    if request.param == "pgvector":
        request.getfixturevalue("pg_url")  # skip early, with the reason
    yield factory
    factory.cleanup()
