"""research/stores.py without a database: choosing the store, the SQL we send, masking, the
no-fallback error and doctor's checks. (The same code against a real Postgres: test_pgvector.py.)"""

import subprocess
import time

import pytest

from shortsmith.research import stores

URL = "postgresql://shortsmith:s3cret-PW_123@127.0.0.1:5433/shortsmith"


# --- choosing the store --------------------------------------------------------------------------------


def test_pgvector_is_the_default_and_chroma_is_a_choice(monkeypatch):
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    assert stores.selected_backend() == "pgvector"
    monkeypatch.setenv("VECTOR_STORE", " Chroma ")
    assert stores.selected_backend() == "chroma"


def test_an_unknown_store_is_an_error_not_a_guess(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "pgvectr")
    with pytest.raises(ValueError, match="VECTOR_STORE='pgvectr'; it must be one of: pgvector, chroma"):
        stores.selected_backend()


def test_table_names_follow_the_collection_names():
    assert stores.table_name("kb-gemini-embedding-2-768") == "kb_gemini_embedding_2_768"
    assert stores.table_name("kb-text-embedding-3-small-1536") == "kb_text_embedding_3_small_1536"
    for bad in ('kb"; DROP TABLE x; --', "kb with space", "1kb", "k" * 64):
        with pytest.raises(ValueError):
            stores.table_name(bad)


# --- no quiet fallback -------------------------------------------------------------------------------------


def test_no_database_url_is_an_error_with_the_next_step(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(stores.StoreUnavailable) as e:
        stores.open_store("kb-fake-64", 64, "fake", backend="pgvector")
    assert e.value.message == "VECTOR_STORE=pgvector, but DATABASE_URL is not set"
    assert ".env.example" in e.value.hint and "docker compose up -d db" in e.value.hint and "VECTOR_STORE=chroma" in e.value.hint
    assert str(e.value).startswith(e.value.message + ". Next: ")  # one line for an MCP client, which shows only str(e)


def test_a_database_that_is_not_there_fails_fast_and_never_shows_the_password():
    url = "postgresql://shortsmith:not-the-real-one@127.0.0.1:1/shortsmith"  # port 1: nothing listens there
    start = time.perf_counter()
    with pytest.raises(stores.StoreUnavailable) as e:
        stores.PgVectorStore("kb_fake_64", 64, url)
    assert time.perf_counter() - start < stores.CONNECT_TIMEOUT_S + 2
    assert e.value.message.startswith("can't connect to the pgvector database at 127.0.0.1:1/shortsmith (")
    assert "not-the-real-one" not in str(e.value) and e.value.hint == stores.NEXT_STEP


# --- the SQL -------------------------------------------------------------------------------------------------


def test_the_top_k_query_is_parameterised():
    query, params = stores.topk_sql("kb_x", "VECTOR", 5, "brand", "anor-va-dolchin")
    assert query.as_string(None) == ('SELECT id, kind, title, group_name, text, embedding <=> %s AS distance FROM "kb_x" '
                                     "WHERE kind = %s AND brand = %s ORDER BY distance LIMIT %s")
    assert params == ["VECTOR", "brand", "anor-va-dolchin", 5]


def test_no_filter_means_no_where_and_one_filter_means_one_condition():
    assert stores.topk_sql("kb_x", "V", 3, None, None)[0].as_string(None).endswith('FROM "kb_x"  ORDER BY distance LIMIT %s')
    query, params = stores.topk_sql("kb_x", "V", 3, "hook", None)
    assert "WHERE kind = %s ORDER BY" in query.as_string(None) and params == ["V", "hook", 3]


def test_what_a_user_types_never_becomes_sql():
    evil = "x' OR '1'='1"
    query, params = stores.topk_sql("kb_x", "V", 5, "brand", evil)
    assert evil not in query.as_string(None) and params[2] == evil  # a value, sent apart from the SQL
    assert stores.topk_sql('we"ird', "V", 1, None, None)[0].as_string(None).count('"we""ird"') == 1  # a quoted identifier


# --- what doctor prints ----------------------------------------------------------------------------------------


def test_masked_urls_hide_only_the_password():
    assert stores.masked(URL) == "postgresql://shortsmith:***@127.0.0.1:5433/shortsmith"
    assert stores.masked("postgresql://localhost/db") == "postgresql://localhost/db"
    assert stores.where_it_is(URL) == "127.0.0.1:5433/shortsmith"


def test_docker_engine_states(monkeypatch):
    monkeypatch.setattr(stores.shutil, "which", lambda name: None)
    assert stores.docker_engine() == (None, "docker is not on PATH")
    monkeypatch.setattr(stores.shutil, "which", lambda name: "C:/fake/docker.exe")

    def answer(code, out):
        return lambda *a, **k: subprocess.CompletedProcess(a, code, stdout=out, stderr="")

    monkeypatch.setattr(stores.subprocess, "run", answer(0, "29.4.0\n"))
    assert stores.docker_engine() == (True, "engine 29.4.0")
    monkeypatch.setattr(stores.subprocess, "run", answer(1, ""))  # "failed to connect to the docker API at npipe:..."
    assert stores.docker_engine() == (False, "the engine is not running (start Docker Desktop)")

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("docker", 10)

    monkeypatch.setattr(stores.subprocess, "run", hang)
    assert stores.docker_engine()[0] is False


def test_status_for_chroma_touches_no_database(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setattr(stores, "connect", lambda *a, **k: pytest.fail("connected"))
    s = stores.status("kb-gemini-embedding-2-768")
    assert (s.backend, s.table, s.reachable) == ("chroma", "kb-gemini-embedding-2-768", None)


def test_status_says_why_the_database_is_unavailable(monkeypatch):
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(stores, "docker_engine", lambda: (False, "the engine is not running (start Docker Desktop)"))
    s = stores.status("kb-gemini-embedding-2-768")
    assert (s.backend, s.table, s.reachable, s.url) == ("pgvector", "kb_gemini_embedding_2_768", False, None)
    assert "DATABASE_URL is not set" in s.error and "docker compose up -d db" in s.hint and s.docker_running is False
