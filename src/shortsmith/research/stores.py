"""Where the knowledge base's vectors live: a small interface and two implementations.

    rag.KnowledgeBase --(hashes / upsert / delete / query)--> VectorStore
                                                              ├── PgVectorStore  Postgres + pgvector (the default)
                                                              └── ChromaStore    a local Chroma folder (chroma/)

Both only store and search: the vectors come from llm.embed() (priced, in the ledger), so a store
never embeds anything. Both return cosine DISTANCE (0 = same direction, 2 = opposite);
KnowledgeBase turns it into similarity = 1 - distance, once, for every store.

Which store: VECTOR_STORE=pgvector (default) or chroma. pgvector needs DATABASE_URL and the database
from docker-compose.yml (start Docker Desktop, then `docker compose up -d db`). If it is selected and
can't be reached, StoreUnavailable says so and what to do: there is NO quiet fallback to Chroma. A
fallback would hide a broken setup and serve whatever stale index happens to be on disk.

psycopg and chromadb are imported inside the classes, so the MCP server and the CLI's read-only
commands don't pay for them (chromadb takes ~2 s to import; psycopg ~0.1 s).
"""

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel

if TYPE_CHECKING:
    import psycopg
    from psycopg import sql

log = logging.getLogger("shortsmith.stores")
T = TypeVar("T")

Backend = Literal["pgvector", "chroma"]
BACKENDS: tuple[Backend, ...] = ("pgvector", "chroma")
DEFAULT_BACKEND: Backend = "pgvector"  # the owner's choice, 2026-10-25
CHROMA_DIR = Path("chroma")  # gitignored; relative to the repo root, like runs/
CONNECT_TIMEOUT_S = 3  # a local database answers in ~0.04 s: 3 s means it's not there

# HNSW, the same graph settings as the Chroma collection had (m = max_neighbors 16, ef_construction 100,
# ef_search 100), so the two stores are built and searched alike. pgvector's defaults are 16 / 64 / 40.
HNSW_M = 16  # links per node: more = better recall, bigger index
HNSW_EF_CONSTRUCTION = 100  # candidates kept while building: more = better graph, slower build
HNSW_EF_SEARCH = 100  # candidates kept while searching: more = better recall, slower query

NEXT_STEP = "start Docker Desktop, then `docker compose up -d db` (in the repo folder), or set VECTOR_STORE=chroma"


class StoreUnavailable(RuntimeError):
    """The selected store can't be used. The message says what is wrong; `hint` says what to do."""

    def __init__(self, message: str, hint: str = NEXT_STEP):
        super().__init__(f"{message}. Next: {hint}")  # one line for the MCP client, which shows only str(e)
        self.message, self.hint = message, hint


class Record(BaseModel):
    """One stored chunk: what rag.Chunk knows, plus the hash of its embedded text and its vector."""

    id: str
    kind: Literal["hook", "brand"]
    title: str
    group: str
    brand: str | None
    source: str
    text: str
    hash: str  # rag.fingerprint(document_text): equal hash = no need to embed again
    embedding: list[float]


class Hit(BaseModel):
    id: str
    kind: Literal["hook", "brand"]
    title: str
    group: str
    text: str
    distance: float  # cosine distance = 1 - cosine similarity


class VectorStore(Protocol):
    """What KnowledgeBase needs from a store, and nothing more."""

    backend: Backend
    name: str  # the collection (Chroma) or table (Postgres): one per embedding model and size

    def count(self) -> int: ...
    def hashes(self) -> dict[str, str]: ...  # id -> hash, for every stored chunk
    def upsert(self, records: list[Record]) -> None: ...  # insert, or replace by id
    def delete(self, ids: list[str]) -> None: ...
    def query(self, vector: list[float], k: int, kind: str | None, brand: str | None) -> list[Hit]: ...  # nearest first
    def records(self) -> list[Record]: ...  # everything, vectors included (for copying between stores)


def selected_backend() -> Backend:
    """VECTOR_STORE, checked: a typo must not quietly pick something else."""
    value = os.environ.get("VECTOR_STORE", DEFAULT_BACKEND).strip().lower() or DEFAULT_BACKEND
    if value not in BACKENDS:
        raise ValueError(f"VECTOR_STORE={value!r}; it must be one of: {', '.join(BACKENDS)}")
    return value  # type: ignore[return-value]


def table_name(collection: str) -> str:
    """kb-gemini-embedding-2-768 -> kb_gemini_embedding_2_768: underscores need no quoting in psql."""
    name = collection.replace("-", "_")
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", name):  # Postgres cuts identifiers at 63 bytes
        raise ValueError(f"{collection!r} can't be a table name")
    return name


def open_store(collection: str, dimensions: int, model: str, backend: Backend | None = None) -> VectorStore:
    """The selected store for one embedding model and size. Never falls back to another one."""
    backend = backend or selected_backend()
    if backend == "chroma":
        return ChromaStore(collection, dimensions, model, CHROMA_DIR)
    return PgVectorStore(table_name(collection), dimensions)


# --- Chroma ------------------------------------------------------------------------------------------


class ChromaStore:
    """One Chroma collection (Phase 5's store, moved here unchanged). path=None: in memory (tests)."""

    backend: Backend = "chroma"

    def __init__(self, collection: str, dimensions: int, model: str, path: Path | None = CHROMA_DIR):
        import chromadb
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False)
        client = (chromadb.PersistentClient(path=str(path), settings=settings) if path
                  else chromadb.EphemeralClient(settings=settings))
        self.collection = client.get_or_create_collection(
            collection,
            embedding_function=None,  # we pass vectors in; Chroma must never embed (or download a model)
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": model, "dimensions": dimensions},
        )
        self.name = self.collection.name

    def count(self) -> int:
        return self.collection.count()

    def hashes(self) -> dict[str, str]:
        stored = self.collection.get(include=["metadatas"])
        return {i: (m or {}).get("hash", "") for i, m in zip(stored["ids"], stored["metadatas"] or [])}

    def upsert(self, records: list[Record]) -> None:
        if records:
            self.collection.upsert(  # insert or replace by id: no duplicates, ever
                ids=[r.id for r in records],
                embeddings=[r.embedding for r in records],
                documents=[r.text for r in records],
                metadatas=[{"kind": r.kind, "title": r.title, "group": r.group, "brand": r.brand or "",
                            "source": r.source, "hash": r.hash} for r in records],
            )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def query(self, vector: list[float], k: int, kind: str | None, brand: str | None) -> list[Hit]:
        filters: list[dict[str, Any]] = [{"kind": kind}] if kind else []
        if brand:
            filters.append({"brand": brand})
        where = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)
        found = self.collection.query(query_embeddings=[vector], n_results=k, where=where,
                                      include=["documents", "metadatas", "distances"])
        return [Hit(id=i, kind=m["kind"], title=m["title"], group=m["group"], text=doc, distance=distance)
                for i, doc, m, distance in zip(found["ids"][0], found["documents"][0], found["metadatas"][0],
                                               found["distances"][0])]

    def records(self) -> list[Record]:
        stored = self.collection.get(include=["documents", "metadatas", "embeddings"])
        return sorted((Record(id=i, kind=m["kind"], title=m["title"], group=m["group"], brand=m["brand"] or None,
                              source=m["source"], text=doc, hash=m["hash"], embedding=[float(x) for x in e])
                       for i, doc, m, e in zip(stored["ids"], stored["documents"], stored["metadatas"],
                                               stored["embeddings"])), key=lambda r: r.id)


# --- Postgres + pgvector -----------------------------------------------------------------------------------


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise StoreUnavailable("VECTOR_STORE=pgvector, but DATABASE_URL is not set",
                               "add DATABASE_URL to .env (see .env.example), start Docker Desktop, then "
                               "`docker compose up -d db`; or set VECTOR_STORE=chroma")
    return url


def masked(url: str) -> str:
    """postgresql://shortsmith:***@127.0.0.1:5433/shortsmith: safe to print (doctor, errors)."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    return parts._replace(netloc=f"{parts.username}:***@{host}{f':{parts.port}' if parts.port else ''}").geturl()


def where_it_is(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname}:{parts.port or 5432}/{parts.path.lstrip('/')}"


def connect(url: str | None = None, vectors: bool = True) -> "psycopg.Connection":
    """A connection with the vector type known to psycopg, or StoreUnavailable (never a fallback).
    vectors=False: a plain connection that changes nothing (doctor)."""
    import psycopg
    from pgvector.psycopg import register_vector

    url = url or database_url()
    try:
        conn = psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S, autocommit=True,
                               application_name="shortsmith")
    except psycopg.OperationalError as e:
        reason = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        raise StoreUnavailable(f"can't connect to the pgvector database at {where_it_is(url)} ({reason[:160]})") from None
    if not vectors:
        return conn
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # idempotent; our user owns the database
        register_vector(conn)  # numpy arrays <-> the vector type, in binary
    except Exception:
        conn.close()
        raise
    return conn


def topk_sql(table: str, vector: Any, k: int, kind: str | None, brand: str | None) -> tuple["sql.Composed", list[Any]]:
    """The filtered nearest-neighbour query and its parameters. Only fixed SQL pieces are joined
    together; the table goes in as a quoted identifier and every value as a parameter (%s), so
    nothing a user types can change the SQL. ORDER BY is the distance with a LIMIT: the shape an
    HNSW index can answer (a second sort key would stop that)."""
    from psycopg import sql

    conditions, params = [], []
    if kind:
        conditions.append(sql.SQL("kind = %s"))
        params.append(kind)
    if brand:
        conditions.append(sql.SQL("brand = %s"))
        params.append(brand)
    where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("")
    query = sql.SQL(
        "SELECT id, kind, title, group_name, text, embedding <=> %s AS distance FROM {table} {where} "
        "ORDER BY distance LIMIT %s"
    ).format(table=sql.Identifier(table), where=where)
    return query, [vector, *params, k]


class PgVectorStore:
    """One table per embedding model and size, in Postgres with pgvector.

    The table is made on first use: one row per chunk, the vector as vector(dimensions), an HNSW
    index for cosine distance. `table` can be any name (the tests use throwaway tables)."""

    backend: Backend = "pgvector"

    def __init__(self, table: str, dimensions: int, url: str | None = None):
        self.name = table
        self.dimensions = dimensions
        self.url = url or database_url()
        self._conn: psycopg.Connection | None = connect(self.url)  # fails here, loudly, if the DB is down
        self.ensure_table()

    def _run(self, work: Callable[["psycopg.Connection"], T]) -> T:
        """work(connection) in one transaction. If the connection was lost since the last call (the
        database restarted), connect again and run it once more: every operation here is safe to
        repeat (reads, upserts by id, deletes by id). Still unreachable: StoreUnavailable."""
        import psycopg

        for attempt in (1, 2):
            if self._conn is None or self._conn.closed or self._conn.broken:
                self._conn = connect(self.url)  # StoreUnavailable if the database is down
            try:
                with self._conn.transaction():
                    return work(self._conn)
            except psycopg.OperationalError as e:
                if not self._conn.broken:  # a real error in the statement, not a lost connection
                    raise
                self._conn = None
                if attempt == 2:
                    raise StoreUnavailable(f"lost the connection to the pgvector database at {where_it_is(self.url)} "
                                           f"twice ({str(e).strip()[:160]})") from None
                log.warning("pgvector: the connection was lost (%s); connecting again", str(e).strip()[:120])
        raise AssertionError("unreachable")

    def _sql(self, template: str) -> "sql.Composed":
        from psycopg import sql

        return sql.SQL(template).format(table=sql.Identifier(self.name), index=sql.Identifier(f"{self.name}_hnsw"))

    def ensure_table(self) -> None:
        def create(conn: "psycopg.Connection") -> None:
            conn.execute(self._sql(
                "CREATE TABLE IF NOT EXISTS {table} ("
                " id text PRIMARY KEY,"  # rag.Chunk.id: stable across rebuilds, so upsert replaces instead of duplicating
                " kind text NOT NULL CHECK (kind IN ('hook', 'brand')),"
                " title text NOT NULL,"
                " group_name text NOT NULL,"  # GROUP is an SQL keyword
                " brand text,"  # NULL for hook patterns
                " source text NOT NULL,"
                " text text NOT NULL,"
                " hash text NOT NULL,"
                f" embedding vector({int(self.dimensions)}) NOT NULL,"  # an int, never user text: the size is checked on every insert
                " updated_at timestamptz NOT NULL DEFAULT now())"))
            conn.execute(self._sql(
                "CREATE INDEX IF NOT EXISTS {index} ON {table} USING hnsw (embedding vector_cosine_ops)"
                f" WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"))
            existing = conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
                [self.name]).fetchone()
            if existing and existing[0] != self.dimensions:  # vector(n) keeps n in atttypmod
                raise ValueError(f"table {self.name} holds {existing[0]}-number vectors, not {self.dimensions}")

        self._run(create)

    def count(self) -> int:
        return self._run(lambda conn: conn.execute(self._sql("SELECT count(*) FROM {table}")).fetchone()[0])

    def hashes(self) -> dict[str, str]:
        return self._run(lambda conn: dict(conn.execute(self._sql("SELECT id, hash FROM {table}")).fetchall()))

    def upsert(self, records: list[Record]) -> None:
        import numpy as np

        if not records:
            return
        rows = [(r.id, r.kind, r.title, r.group, r.brand, r.source, r.text, r.hash,
                 np.asarray(r.embedding, dtype=np.float32)) for r in records]

        def write(conn: "psycopg.Connection") -> None:
            with conn.cursor() as cur:
                cur.executemany(self._sql(  # one transaction, sent as one pipeline
                    "INSERT INTO {table} (id, kind, title, group_name, brand, source, text, hash, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT (id) DO UPDATE SET kind = EXCLUDED.kind, title = EXCLUDED.title,"
                    " group_name = EXCLUDED.group_name, brand = EXCLUDED.brand, source = EXCLUDED.source,"
                    " text = EXCLUDED.text, hash = EXCLUDED.hash, embedding = EXCLUDED.embedding, updated_at = now()"),
                    rows)

        self._run(write)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._run(lambda conn: conn.execute(self._sql("DELETE FROM {table} WHERE id = ANY(%s)"), [ids]))

    def query(self, vector: list[float], k: int, kind: str | None, brand: str | None) -> list[Hit]:
        import numpy as np

        query, params = topk_sql(self.name, np.asarray(vector, dtype=np.float32), k, kind, brand)

        def search(conn: "psycopg.Connection") -> list[tuple]:
            # SET LOCAL lasts until this transaction ends: nothing leaks to the next user of the connection.
            # iterative_scan: if a WHERE filter throws away most of the index's ef_search candidates, keep
            # scanning the graph until k rows pass (pgvector >= 0.8), in exact distance order.
            conn.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}; SET LOCAL hnsw.iterative_scan = strict_order")
            return conn.execute(query, params).fetchall()

        return [Hit(id=i, kind=kd, title=t, group=g, text=txt, distance=float(d)) for i, kd, t, g, txt, d in self._run(search)]

    def records(self) -> list[Record]:
        rows = self._run(lambda conn: conn.execute(self._sql(
            "SELECT id, kind, title, group_name, brand, source, text, hash, embedding FROM {table} ORDER BY id")).fetchall())
        return [Record(id=i, kind=kd, title=t, group=g, brand=b, source=s, text=txt, hash=h,
                       embedding=e.to_list())  # pgvector-python 0.5 loads a vector as a Vector object
                for i, kd, t, g, b, s, txt, h, e in rows]

    def drop(self) -> None:
        """Delete the table (tests only: their throwaway tables)."""
        self._run(lambda conn: conn.execute(self._sql("DROP TABLE IF EXISTS {table}")))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# --- for `shortsmith doctor` -------------------------------------------------------------------------


class StoreStatus(BaseModel):
    backend: Backend
    table: str
    url: str | None = None  # masked: never the password
    docker: str | None = None  # the engine's version, or why there is none
    docker_running: bool | None = None
    reachable: bool | None = None
    error: str | None = None
    hint: str | None = None  # what to do about the error
    postgres: str | None = None
    pgvector: str | None = None
    rows: int | None = None  # None: the table isn't there yet
    seconds: float = 0.0


def docker_engine() -> tuple[bool | None, str]:
    """(running?, what docker says). None: docker isn't installed. A few seconds at most."""
    if shutil.which("docker") is None:
        return None, "docker is not on PATH"
    try:
        done = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                              timeout=10, errors="replace")
    except subprocess.TimeoutExpired:
        return False, "`docker info` did not answer in 10 s"
    if done.returncode == 0 and done.stdout.strip():
        return True, f"engine {done.stdout.strip()}"
    return False, "the engine is not running (start Docker Desktop)"


def status(collection: str) -> StoreStatus:
    """What doctor prints about the vector store. Reads only: creates no table, embeds nothing."""
    start = time.perf_counter()
    backend = selected_backend()
    if backend == "chroma":
        return StoreStatus(backend=backend, table=collection, seconds=round(time.perf_counter() - start, 2))
    table = table_name(collection)
    s = StoreStatus(backend=backend, table=table)
    s.docker_running, s.docker = docker_engine()
    try:
        url = database_url()
        s.url = masked(url)
        with connect(url, vectors=False) as conn:
            s.reachable = True
            s.postgres = conn.execute("SHOW server_version").fetchone()[0]
            extension = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            s.pgvector = extension[0] if extension else None  # None: not created yet (the first store use does it)
            if conn.execute("SELECT to_regclass(%s)", [table]).fetchone()[0] is not None:
                from psycopg import sql

                s.rows = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
    except StoreUnavailable as e:
        s.reachable, s.error, s.hint = False, e.message, e.hint
    s.seconds = round(time.perf_counter() - start, 2)
    return s
