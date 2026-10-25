"""Integration tests against the real Postgres + pgvector (marker `pgvector`; see conftest.py for how to
run them). Each test makes its own throwaway tables (test_<random>_<name>) and drops them; the real
index is never touched. The behaviour tests in test_rag.py also run here, through store_factory."""

import math
import random
from urllib.parse import urlsplit

import numpy as np
import pytest
from psycopg import sql

from shortsmith import llm
from shortsmith.research import migrate, rag, stores

pytestmark = pytest.mark.pgvector
DIMS = 16


def record(i: int, vector: list[float], kind: str = "hook", brand: str | None = None, text: str = "") -> stores.Record:
    return stores.Record(id=f"{kind}/{i:03d}", kind=kind, title=f"t{i}", group="g", brand=brand, source="s.md",
                         text=text or f"text {i}", hash=f"h{i}", embedding=vector)


def updated_at(store: stores.PgVectorStore, id: str):
    return store._conn.execute(sql.SQL("SELECT updated_at FROM {} WHERE id = %s").format(sql.Identifier(store.name)),
                               [id]).fetchone()[0]


def unit(vector: list[float]) -> list[float]:
    n = math.hypot(*vector)
    return [x / n for x in vector]


@pytest.fixture
def pg(request, tmp_path, pg_url):
    from conftest import StoreFactory

    factory = StoreFactory("pgvector", tmp_path, request, dimensions=DIMS)
    yield factory
    factory.cleanup()


def test_the_table_has_the_vector_column_and_a_cosine_hnsw_index(pg):
    store = pg.open("schema")
    conn = store._conn
    assert conn.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid = %s::regclass "
                        "AND attname = 'embedding'", [store.name]).fetchone()[0] == f"vector({DIMS})"
    indexdef, options = conn.execute(
        "SELECT pg_get_indexdef(i.indexrelid), c.reloptions FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE i.indrelid = %s::regclass AND c.relname = %s", [store.name, f"{store.name}_hnsw"]).fetchone()
    assert "USING hnsw (embedding vector_cosine_ops)" in indexdef
    assert sorted(options) == [f"ef_construction={stores.HNSW_EF_CONSTRUCTION}", f"m={stores.HNSW_M}"]
    assert conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0] >= "0.8"


def test_vectors_of_another_size_are_refused(pg):
    pg.open("size", dimensions=DIMS)
    with pytest.raises(ValueError, match=f"holds {DIMS}-number vectors, not 8"):
        pg.open("size", dimensions=8)


def test_an_upsert_replaces_by_id_and_a_delete_removes(pg):
    store = pg.open("upsert")
    store.upsert([record(1, [1.0] + [0.0] * (DIMS - 1)), record(2, [0.0, 1.0] + [0.0] * (DIMS - 2))])
    first = updated_at(store, "hook/001")
    store.upsert([record(1, [0.0, 0.0, 1.0] + [0.0] * (DIMS - 3), text="edited")])
    rows = {r.id: r for r in store.records()}
    assert store.count() == 2 and rows["hook/001"].text == "edited" and rows["hook/001"].embedding[2] == 1.0
    assert updated_at(store, "hook/001") > first
    store.delete(["hook/002", "never-there"])
    assert store.hashes() == {"hook/001": "h1"}


def random_records(n: int = 60, seed: int = 3) -> list[stores.Record]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind, brand = ("brand", ["anor", "samoviy"][i % 2]) if i % 5 == 0 else ("hook", None)
        out.append(record(i, [rng.gauss(0, 1) for _ in range(DIMS)], kind, brand))
    return out


def test_scores_match_chroma_and_the_formula_on_hand_made_vectors(pg, tmp_path):
    """The same records in both stores, the same queries and filters: the same ids in the same order,
    and distances equal to 1 - cos to float32 precision (both stores keep 4-byte floats)."""
    records = random_records()
    pgs = pg.open("parity")
    chroma = stores.ChromaStore("kb-parity", DIMS, "fake", tmp_path / "chroma")
    for store in (pgs, chroma):
        store.upsert(records)
    rng = random.Random(9)
    for _ in range(10):
        q = [rng.gauss(0, 1) for _ in range(DIMS)]
        for kind, brand, k in [(None, None, 7), ("hook", None, 5), ("brand", None, 5), ("brand", "samoviy", 3)]:
            a, b = pgs.query(q, k, kind, brand), chroma.query(q, k, kind, brand)
            pool = [r for r in records if (not kind or r.kind == kind) and (not brand or r.brand == brand)]
            exact = sorted(pool, key=lambda r: 1 - sum(x * y for x, y in zip(unit(q), unit(r.embedding))))[:k]
            assert [h.id for h in a] == [h.id for h in b] == [r.id for r in exact]
            assert [h.distance for h in a] == pytest.approx([h.distance for h in b], abs=1e-6)
            assert all(h.kind == (kind or h.kind) for h in a)


def test_a_selective_filter_still_gets_k_rows_when_the_index_is_used(pg):
    """300 rows: 290 hooks right next to the query, 10 brand rows far from it. An HNSW scan looks at
    ef_search (100) candidates, all of them hooks, so WHERE kind = 'brand' would leave nothing. The
    store's query keeps scanning (hnsw.iterative_scan = strict_order) and gets the exact top 5."""
    rng = random.Random(1)
    near = [1.0] + [0.0] * (DIMS - 1)
    records = [record(i, [x + rng.gauss(0, 0.05) for x in near]) for i in range(290)]
    records += [record(i, [rng.gauss(0, 0.05), 1.0] + [rng.gauss(0, 0.05) for _ in range(DIMS - 2)], "brand", "b")
                for i in range(290, 300)]
    store = pg.open("filter")
    store.upsert(records)
    conn = store._conn
    conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(store.name)))
    conn.execute("SET enable_seqscan = off")  # this test's connection only (the store's): make the planner use the index
    query, params = stores.topk_sql(store.name, np.asarray(near, dtype=np.float32), 5, "brand", None)
    for iterative in ("off", "strict_order"):
        with conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {stores.HNSW_EF_SEARCH}; SET LOCAL hnsw.iterative_scan = {iterative}")
            plan = "\n".join(r[0] for r in conn.execute(sql.SQL("EXPLAIN ") + query, params))
            rows = conn.execute(query, params).fetchall()
        assert f"Index Scan using {store.name}_hnsw" in plan
        if iterative == "off":
            assert len(rows) < 5  # the problem is real here: a plain index scan gives too few rows
    hits = store.query(near, 5, "brand", None)
    exact = sorted(records[290:], key=lambda r: 1 - sum(x * y for x, y in zip(unit(near), unit(r.embedding))))[:5]
    assert [h.id for h in hits] == [r.id for r in exact]


def test_a_lost_connection_is_opened_again(pg, pg_url, caplog):
    store = pg.open("reconnect")
    store.upsert(random_records(5))
    with stores.connect(pg_url, vectors=False) as admin:  # what a database restart does to our session
        admin.execute("SELECT pg_terminate_backend(%s)", [store._conn.info.backend_pid])
    assert store.count() == 5
    assert "connection was lost" in caplog.text


def test_the_wrong_password_is_an_error_without_the_password(pg_url):
    u = urlsplit(pg_url)
    wrong = u._replace(netloc=f"{u.username}:wrong-password-xyz@{u.hostname}:{u.port}").geturl()
    with pytest.raises(stores.StoreUnavailable) as e:
        stores.PgVectorStore("kb_never", DIMS, wrong)
    assert "password authentication failed" in e.value.message and "wrong-password-xyz" not in str(e.value)


def test_copying_from_chroma_needs_no_embedding_and_the_next_sync_embeds_nothing(pg, tmp_path):
    """The migration in miniature: vectors are portable between stores, the hashes travel with them."""
    folder = tmp_path / "hooks"
    folder.mkdir()
    (folder / "curiosity.md").write_text("# Curiosity\n\n## Open loop\n\nA strange fact.\n\n## Guess the price\n\nHow much?\n",
                                         encoding="utf-8")

    def embed(texts: list[str]) -> llm.EmbedResult:
        rng = random.Random(len(texts))
        return llm.EmbedResult(vectors=[[rng.gauss(0, 1) for _ in range(DIMS)] for _ in texts], model="fake",
                               dimensions=DIMS, prompt_tokens=1, tokens_estimated=True, cost_usd=0.0, latency_s=0.0)

    chroma = stores.ChromaStore("kb-copy", DIMS, "fake", tmp_path / "chroma")
    rag.KnowledgeBase(data_dir=folder, store=chroma, model="fake", dimensions=DIMS, embed=embed).sync()
    target = pg.open("copy")
    report = migrate.copy_index(chroma, target)
    assert (report.copied, report.target_rows, report.hashes_match, report.max_vector_difference) == (2, 2, True, 0.0)
    assert migrate.copy_index(chroma, target).target_rows == 2  # idempotent

    def refuse(texts: list[str]) -> llm.EmbedResult:
        raise AssertionError(f"embedded {texts}")

    assert rag.KnowledgeBase(data_dir=folder, store=target, model="fake", dimensions=DIMS, embed=refuse).sync().unchanged == 2
