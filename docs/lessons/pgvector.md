# Side step — From Chroma to pgvector: vector search inside Postgres

> Read at your own pace, ask anything. Parts 0–10 are the concepts. Parts 11–16 are what was really built and
> measured (from `docs/lessons/pgvector-build-notes.md`, commit 347b4fe): the compose file, the schema, the
> migration, the parity proof, real EXPLAIN plans, errors, tests.

---

## 0. Why we switched (and what it will *not* change)

Phase 5 stored the 79 hook and brand chunks in **Chroma**: a vector database that runs inside our Python
process. It worked well (the right pattern came first 88% of the time). We switched to **pgvector** for
production reasons, not quality:
- **One database.** Phase 8–9 will need Postgres anyway (users, runs, costs). With pgvector the knowledge base
  lives in the same database: one thing to run, back up, secure and query.
- **SQL.** Vectors become one more column, so you can join them, filter them and wrap them in transactions with
  everything else, using the SQL you already know.
- **It's what many companies run**, and the most defensible interview answer for "vectors at our scale".

**What it won't change:** retrieval quality. The same model and the same chunks with the same metric give the
same scores. The build has to *prove* that (part 13), not assume it.

---

## 1. Postgres extensions, and the `vector` type

Postgres can be extended with new types, functions and index methods. **pgvector** is such an extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;    -- once per database
```
After that, Postgres has:
- a new column type, **`vector(768)`**: an array of 768 4-byte floats (about 3 KB per row);
- **distance operators**:

  | Operator | Distance | Use with |
  |---|---|---|
  | `<->` | Euclidean (L2) | |
  | `<#>` | **negative** inner product (negative, so that "smaller = closer" for every operator) | normalised vectors |
  | **`<=>`** | **cosine distance** = 1 − cosine similarity | **ours** (Phase 5 used cosine) |
  | `<+>` | L1 (taxicab) | |

- **index types** for fast approximate search: HNSW and IVFFlat (part 5);
- related types for compression: `halfvec` (2-byte floats) and `bit` (binary vectors).

**Similarity from distance:** Chroma gave us `similarity = 1 − cosine distance`. In SQL it's the same formula:
`1 - (embedding <=> query)`. That's why the scores can match to the last decimal.

---

## 2. Docker, just enough for this

You'll run Postgres in a **container**, not install it on Windows.

| Word | Meaning | Ours |
|---|---|---|
| **image** | a packaged, read-only filesystem with a program in it | `pgvector/pgvector:<tag>`: official Postgres with pgvector already built in |
| **container** | a running instance of an image, isolated from your PC | the `db` service |
| **volume** | storage that lives **outside** the container, so data survives restarts and image updates | the database files |
| **port mapping** | connects a port on your PC to a port in the container | `127.0.0.1:5433 → 5432` |
| **docker compose** | a YAML file that describes services, so one command starts them | `docker-compose.yml` in the repo |
| **healthcheck** | a command Docker runs to know when the service is really ready | `pg_isready` |

A compose file of the kind the build will write (the real one is in part 11):
```yaml
services:
  db:
    image: pgvector/pgvector:0.8.1-pg17        # an exact tag, never "latest": the same version every time
    environment:
      POSTGRES_USER: shortsmith
      POSTGRES_PASSWORD: ${SHORTSMITH_DB_PASSWORD}   # read from .env, never written in the file
      POSTGRES_DB: shortsmith
    ports:
      - "127.0.0.1:5433:5432"                 # only this PC can connect; 5433 so it can't clash with your Windows Postgres on 5432
    volumes:
      - pgdata:/var/lib/postgresql/data       # the data lives in a named volume
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U shortsmith -d shortsmith"]
      interval: 5s
      retries: 10
volumes:
  pgdata:
```
**Why `127.0.0.1:`:** without it, Docker publishes the port on every network your PC is on, so anyone on the
same Wi-Fi could try to connect. Binding to 127.0.0.1 makes it local-only.

**The commands** (terminal, repo root):
```bash
docker compose up -d db
```
```bash
docker compose ps
```
```bash
docker compose logs db
```
```bash
docker compose stop db
```
- `up -d` starts it in the background (`-d` = detached), pulling the image the first time.
- `ps` shows whether it's running and **healthy**.
- `logs` shows what Postgres printed.
- `stop` stops it, and the data stays. `docker compose start db` brings it back.
- **Never `docker compose down -v`**: `-v` deletes the volumes, which means the database. (The same rule protects
  Yukdosh's server.)

**Docker Desktop must be running** for any of this: it provides the Docker engine on Windows.

---

## 3. Connecting from Python

A **connection string** packs everything into one line:
```
postgresql://shortsmith:<password>@127.0.0.1:5433/shortsmith
            └─ user ──┘ └password┘ └─ host ─┘ └port┘ └ database
```
It lives in `.env` as `DATABASE_URL`, never in code or in a commit.

The driver is **psycopg 3**, and the **`pgvector` Python package** teaches it the `vector` type:
```python
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

with psycopg.connect(database_url) as conn:
    register_vector(conn)                          # now numpy arrays <-> vector columns
    rows = conn.execute(
        """SELECT id, title, 1 - (embedding <=> %s) AS similarity
             FROM kb_chunks
            WHERE kind = %s
            ORDER BY embedding <=> %s
            LIMIT %s""",
        (np.array(query_vec), "hook", np.array(query_vec), 5),
    ).fetchall()
```
- **`%s` placeholders, never f-strings.** The driver sends values separately from the SQL text, so a topic like
  `'; DROP TABLE kb_chunks; --` is just text. That's **SQL injection** prevention, and an interview classic.
- `ORDER BY embedding <=> %s LIMIT k` **is** top-k retrieval: the nearest k rows by cosine distance.
- (Real apps reuse connections with a **pool** instead of opening one per query. At our volume one connection
  per call is fine.)

---

## 4. Schema and upsert: what Chroma did for us, in SQL

Chroma's `upsert`, `delete` and `get` become plain SQL:
```sql
CREATE TABLE IF NOT EXISTS kb_chunks (
    id        text PRIMARY KEY,        -- 'curiosity/open-loop', 'brands/anor-va-dolchin/visual-style'
    kind      text NOT NULL,           -- 'hook' or 'brand'
    grp       text,                    -- the file or the brand
    title     text NOT NULL,
    text      text NOT NULL,
    hash      text NOT NULL,           -- the idempotent sync: only changed text is re-embedded
    embedding vector(768) NOT NULL
);

INSERT INTO kb_chunks (id, kind, grp, title, text, hash, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE                    -- "upsert": insert, or replace the row with that id
   SET kind = EXCLUDED.kind, title = EXCLUDED.title, text = EXCLUDED.text,
       hash = EXCLUDED.hash, embedding = EXCLUDED.embedding;
```
- `ON CONFLICT (id) DO UPDATE` is how Postgres says "insert or replace", so re-running a sync can't create
  duplicates.
- **Vectors from different models must never mix.** Chroma kept them apart by collection name
  (`kb-gemini-embedding-2-768`). In SQL, that's either one table per model and size, or a `collection` column.
  (`vector(768)` fixes the size per column anyway.) The build decides and explains (part 12).
- A whole sync can run in **one transaction**: either every change lands, or none does.

---

## 5. Indexes: HNSW and IVFFlat inside Postgres

Without an index, `ORDER BY embedding <=> q LIMIT 5` compares the query with **every row**. That's a sequential
scan, which is **exact** search (brute force, recall 100%). Fine for 79 rows, too slow for millions. pgvector
offers two approximate indexes:

**HNSW** (the layered graph from the widget you played with):
```sql
CREATE INDEX ON kb_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
SET hnsw.ef_search = 40;     -- per session/query: more = better recall, slower
```
- **`vector_cosine_ops`**: the index must be built for the **same operator** you query with (`<=>`). An index
  built for L2 won't be used by a cosine query.
- `m` = links per node (default 16); `ef_construction` = how carefully the graph is built (default 64);
  `hnsw.ef_search` = how many candidates a query keeps (default 40).
- Good recall and speed; uses memory; building it is slower than IVFFlat.

**IVFFlat:** the vectors are split into `lists` clusters, and a query searches only `ivfflat.probes` of them.
It's faster to build and smaller, but recall depends on `probes`. It should be built **after** the data is loaded,
because the clusters are learned from it.

**Index size limits:** an HNSW or IVFFlat index on `vector` supports up to **2,000 dimensions**; `halfvec` up to
4,000. Our 768 is well inside. (This is one reason we kept 768 instead of Gemini's default 3,072.)

---

## 6. Will Postgres even use the index? `EXPLAIN ANALYZE`

Postgres has a **query planner**: before running a query it estimates the cost of each possible plan and picks
the cheapest. You can see its choice:
```sql
EXPLAIN ANALYZE
SELECT id FROM kb_chunks ORDER BY embedding <=> '[...]' LIMIT 5;
```
- **`Seq Scan`** = it read every row (exact search).
- **`Index Scan using ..._hnsw_idx`** = it used the HNSW graph (approximate).
- `ANALYZE` runs the query for real and shows actual times, not just estimates.

**Prediction for us:** with 79 rows (about 1 MB), the planner will probably choose the **sequential scan**:
reading 79 small rows is cheaper than walking a graph. That's correct behaviour, not a bug. The build will show
both plans (forcing the index with `SET enable_seqscan = off`) with real timings. **Being able to read a plan and
explain why the index wasn't used** is a real interview skill.

---

## 7. Filters + approximate search: the classic trap

Our queries filter: `WHERE kind = 'hook'`, or `WHERE grp = 'anor-va-dolchin'`. With an HNSW index, Postgres by
default:
1. walks the graph and gets the `ef_search` nearest candidates (say 40);
2. **then** applies the `WHERE`.

If only a few of those 40 match the filter, you get **fewer than k results**, even though matching rows exist.
That's the **post-filtering** problem from the interview list.

pgvector 0.8 added **iterative index scans**:
```sql
SET hnsw.iterative_scan = relaxed_order;   -- or strict_order
```
When too few candidates pass the filter, the scan **keeps going** further into the graph until it has enough, up
to a limit. Other tools for the same problem:
- **partial indexes** (`... WHERE kind = 'hook'`);
- **partitioning** by tenant or kind;
- an exact scan when the filter is very selective.

At 79 rows the sequential scan avoids the trap completely. It matters as data grows.

---

## 8. The store interface: swapping the database without touching the rest

Phase 5's `KnowledgeBase` called Chroma directly in 5 places. The build puts a small **interface** in between:
```python
class VectorStore(Protocol):
    def stored_hashes(self) -> dict[str, str]: ...
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def query(self, vector: list[float], k: int, kind: str | None, brand: str | None) -> list[Retrieved]: ...
```
- `KnowledgeBase` (chunking, hash sync, top-k, the `Retrieved` result) talks only to `VectorStore`.
  `ChromaStore` and `PgVectorStore` both implement it. That's the **adapter pattern**, a form of the
  **dependency inversion** you met in Phase 3 (`run_loop` received `call_tool`).
- Tests can use a tiny fake store: fast, offline, no Docker.
- **The choice is explicit:** `VECTOR_STORE=pgvector|chroma`, default pgvector.

**No silent fallback.** If pgvector is selected and the database is down, the program must **fail with a clear
message** ("start Docker Desktop, then `docker compose up -d db`, or set `VECTOR_STORE=chroma`"). It must **not**
quietly use Chroma. You've seen why in Yukdosh: a silent SQLite fallback once made the app run against the wrong
database, and nobody noticed. A fallback that hides a broken dependency turns one clear error into a mystery.

---

## 9. Migration without re-embedding

A vector depends only on **the model, the size, the task format and the text**. None of that changes when the
*storage* changes. So the 79 vectors can be **copied** from Chroma into Postgres: **0 embedding calls, $0, no
free-tier quota**. A `sync()` afterwards must then report "0 to embed": the hashes match.

**Proving it** (part 13):
- the same Phase 5 queries must return **the same ids in the same order with the same scores** from both stores;
- the retrieval eval must give the same **hit@1 0.88 / MRR 0.91**.

That's how real migrations are done: run old and new side by side on the same inputs, compare, and only then
switch over.

---

## 10. Where pgvector stops being enough, and Phase 8–9

**pgvector's limits:**
- **Memory and build time:** a big HNSW index wants to fit in RAM; building it on millions of rows takes time and
  memory (`maintenance_work_mem`).
- **One Postgres server** carries everything: very high query rates or billions of vectors are where dedicated
  engines (Qdrant, Milvus) or managed services win.
- **Compression** helps first: `halfvec` (half the size) or binary vectors with re-ranking.
- A rough rule, from our comparison of vector stores: **pgvector until roughly 10 million vectors, or until heavy
  filtering at high QPS**; then consider a dedicated engine.

**What Phase 8–9 gains from this step:**
- **Users, runs, costs and vectors in one database:**
  ```sql
  SELECT r.title, r.cost FROM runs r JOIN ... WHERE r.user_id = %s
  ```
  joins sit next to vector search.
- **Transactions across everything:** a run and its research saved together, or not at all.
- **One backup** (`pg_dump`) for all of it, and the same Docker setup on the Azure server.
- **Multi-tenancy** later: a `user_id` column plus a filter, or Postgres's row-level security.

---

## 11. Our compose file, commands and setup

**Installed:** the image `pgvector/pgvector:0.8.7-pg17` (PostgreSQL 17.11 + pgvector 0.8.7), psycopg 3.3.6 and
pgvector-python 0.5.1. psycopg's native files loaded fine under Smart App Control, so no version pin was needed.

```yaml
name: shortsmith                         # project name: container shortsmith-db-1, volume shortsmith_pgdata
services:
  db:
    image: pgvector/pgvector:0.8.7-pg17  # exact tag; "pg17" alone would move to each new release
    environment:
      POSTGRES_USER: ${SHORTSMITH_DB_USER:-shortsmith}       # from .env, default "shortsmith"
      POSTGRES_DB: ${SHORTSMITH_DB_NAME:-shortsmith}
      POSTGRES_PASSWORD: ${SHORTSMITH_DB_PASSWORD:?set SHORTSMITH_DB_PASSWORD in .env (see .env.example)}
    ports:
      - "127.0.0.1:5433:5432"            # only this PC; 5433 because Yukdosh's Windows Postgres uses 5432
    volumes:
      - pgdata:/var/lib/postgresql/data  # the data lives in a named volume
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s
    restart: unless-stopped              # comes back with Docker Desktop, unless you stopped it yourself
volumes:
  pgdata:
```
Three small syntax details worth knowing:
- **`${VAR:-default}`** uses a default when the variable isn't set.
- **`${VAR:?message}`** **refuses to start** without the variable and prints that message. That way a forgotten
  password can't silently become an empty one.
- **`$${…}`** is a literal `$` for the container's own shell, so compose doesn't substitute it itself.

**Secrets:** a random 32-character password (`secrets.token_urlsafe(24)`) was **appended** to `.env` as
`SHORTSMITH_DB_PASSWORD` and `DATABASE_URL`, never printed. Careful: `docker compose config` and `docker inspect`
would print the password, so don't run them where others can see your screen.

**Your commands** (terminal, repo root). Start Docker Desktop first.
```bash
docker compose up -d db
```
```bash
docker compose ps
```
```bash
uv run shortsmith doctor
```
```bash
docker compose stop db
```
- `ps` should show `Up … (healthy)   127.0.0.1:5433->5432/tcp`.
- `doctor` now has three new rows: `vector store`, `Docker`, `database` (the URL masked as `shortsmith:***@…`).
- `stop` keeps the data. **Never `down -v`.**

What `doctor` showed live:
```
| vector store | -  | pgvector (the default): table kb_gemini_embedding_2_768 at postgresql://shortsmith:***@127.0.0.1:5433/shortsmith |
| Docker       | ok | engine 29.4.0                                                                                                  |
| database     | ok | PostgreSQL 17.11 (Debian 17.11-1.pgdg12+2), pgvector 0.8.7; 79 rows in kb_gemini_embedding_2_768; 1.02 s       |
```

---

## 12. Our schema, store interface and wiring

### The table
```
                Table "public.kb_gemini_embedding_2_768"
   Column   |           Type           | Nullable | Default
------------+--------------------------+----------+---------
 id         | text                     | not null |            ← PRIMARY KEY, so upsert replaces by id
 kind       | text                     | not null |            ← CHECK: 'hook' or 'brand'
 title      | text                     | not null |
 group_name | text                     | not null |            ← "group" is an SQL keyword
 brand      | text                     |          |            ← NULL for hook patterns
 source     | text                     | not null |
 text       | text                     | not null |
 hash       | text                     | not null |            ← the idempotent sync
 embedding  | vector(768)              | not null |            ← the size is checked on every insert
 updated_at | timestamp with time zone | not null | now()      ← when this chunk was last re-embedded
Indexes:
    "kb_gemini_embedding_2_768_pkey" PRIMARY KEY, btree (id)
    "kb_gemini_embedding_2_768_hnsw" hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='100')
```
- **One table per embedding model and size**, the same idea as Chroma's collection name (D1):
  - an HNSW index needs a fixed size (`vector(768)`), so one shared table couldn't hold a 3,072-number model later;
  - a shared table would need `WHERE collection = …` on **every** query, which is exactly the filtering trap of §7;
  - switching models gets its own table, and retiring one is a single `DROP TABLE`.

  The price is dynamic table names. They're checked against a strict pattern and quoted with
  `psycopg.sql.Identifier`. A test sends `kb"; DROP TABLE x; --` as a table name, and it's refused.
- **The HNSW settings were copied from Chroma's collection** (`m 16`, `ef_construction 100`, `ef_search 100`),
  so "Chroma vs pgvector" compares the stores, not their tuning. pgvector's own defaults are 16 / 64 / 40.

### The query (all values are parameters)
```sql
SELECT id, kind, title, group_name, text, embedding <=> %s AS distance
  FROM "kb_gemini_embedding_2_768"
 WHERE kind = %s AND brand = %s
 ORDER BY distance
 LIMIT %s
-- params: [the query vector as a float32 array, 'brand', 'anor-va-dolchin', 5]
```
- Inside the same transaction: `SET LOCAL hnsw.ef_search = 100` and **`SET LOCAL hnsw.iterative_scan =
  strict_order`** (§14). `SET LOCAL` ends with the transaction, so nothing leaks to the next user of a connection.
  That will matter when Phase 8 adds a connection pool.
- A test sends the brand `x' OR '1'='1` and checks it ends up in the parameters, not in the SQL. That's the SQL
  injection defence, proven.

### The interface (`research/stores.py`)
```python
class VectorStore(Protocol):
    backend: Backend          # "pgvector" | "chroma"
    name: str                 # the table or collection
    def count(self) -> int: ...
    def hashes(self) -> dict[str, str]: ...                    # for sync()
    def upsert(self, records: list[Record]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def query(self, vector, k, kind, brand) -> list[Hit]: ...  # nearest first, cosine DISTANCE
    def records(self) -> list[Record]: ...                     # everything, vectors included (for copying)
```
- **Stores return distance; `KnowledgeBase.nearest()` turns it into similarity in one place**
  (`round(1 - distance, 4)`), so the scores can't drift apart between stores (D5).
- **`KnowledgeBase.sync()` and `retrieve()` are Phase 5's logic, unchanged**: the same chunking, the same hashes,
  the same "embed only new or edited text".
- **`VECTOR_STORE=pgvector` (default) or `chroma`.** A typo is an error, not a guess.
- **No fallback, in three places** (D3):
  - `open_store()` returns the selected store or raises;
  - `preflight()` opens it **before** a researched run starts (no run folder, no LLM call, no ledger line);
  - `doctor` marks it FAIL.
- **Reconnect once, to the same database** (D4, E8): if Docker restarted the database, the next operation
  reconnects and runs again. That's not a fallback, because it's the same store and the same data.
- **Lazy imports:** `import shortsmith.research.rag` loads neither psycopg, numpy nor chromadb, so the CLI and the
  MCP server still start fast.

---

## 13. The migration and the parity proof

### Copying, 0 embedding calls
```
> uv run --env-file .env python -m shortsmith.research.migrate
source chroma:kb-gemini-embedding-2-768: 79 rows | target pgvector:kb_gemini_embedding_2_768
copied 79 | removed 0 | target now 79 rows | hashes match: True | max vector difference 0 | 0.12 s
sync on the target: +0 new, 0 changed, -0 removed, 79 unchanged | embedded 0 tokens
```
- **`max vector difference 0`**: both stores keep 4-byte floats, so the copy is bit-for-bit exact.
- **"0 tokens" is enforced, not just reported:** the sync after the copy used an embed function that **raises if
  it's called**. The ledger had 188 lines before and after.
- **Running it twice gave the same result** (upsert by id): the migration is idempotent.

### The same answers from both stores
The Phase 5 example queries, in Chroma, in pgvector, and with the plain cosine formula in Python ("exact"):

| Query | Top results (score) | Same in all three? |
|---|---|---|
| cozy coffee shop in Tashkent, autumn vibe | 0.6703 POV · 0.6583 The hidden gem · 0.6465 Save the address · 0.6443 Only this month · 0.6347 Walk with me | ✅ |
| Toshkentdagi shinam qahvaxona, kuzgi kayfiyat | 0.6826 POV · 0.6346 Unpopular opinion · 0.6331 The hidden gem · … | ✅ |
| coffee shop: tone, words to use and avoid… (brand anor-va-dolchin) | 0.7905 Words we use and avoid · 0.7715 Voice and tone · 0.7645 Visual style · … | ✅ |
| guess the price of melons at the bazaar (no filter) | 0.7071 Guess the price · 0.6516 The odd object · … | ✅ |

- **These are exactly the scores Phase 5 printed two days earlier**: the same ids, the same order, the same 4
  decimals.
- **Without rounding**, over 22 queries × 79 rows the largest difference from an exact float64 cosine was **2.2e-7**
  for pgvector (6.4e-7 for Chroma): ordinary 4-byte float arithmetic, invisible after 4 decimals.
- **The retrieval eval through each store:**

| Store | hit@1 | hit@3 | MRR |
|---|---:|---:|---:|
| Phase 5 (plain Python) | 0.88 | 0.94 | 0.91 |
| **Chroma** | 0.88 | 0.94 | **0.914** |
| **pgvector** | **0.88** | 0.94 | **0.914** |

  The same single miss in all of them. **The quality didn't change, as predicted in §0, and now it's proven.**

### `--research` end to end on pgvector
`shortsmith make "tea houses of Samarkand" --lang uz --research --brand samoviy-karvon --simple`: 60 s, 4 calls,
$0.0032 list price ($0 billed). The trends came from the 24-hour cache (0 YouTube units). The hooks and brand rules
came from Postgres. The video was 28.0 s.

---

## 14. EXPLAIN ANALYZE, for real; filtered search

### 79 rows: the planner picks the sequential scan, as predicted
```
Limit  (cost=21.31..21.32 rows=5) (actual time=0.471..0.473 rows=5)
  ->  Sort  (Sort Key: (embedding <=> '[768 numbers]'::vector))   Sort Method: top-N heapsort  Memory: 33kB
        ->  Seq Scan on kb_gemini_embedding_2_768  (cost=0.00..20.16 rows=69) (actual time=0.108..0.401 rows=69)
              Filter: (kind = 'hook'::text)
              Rows Removed by Filter: 10
Execution Time: 0.550 ms
```
**How to read it, from the inside out:**
1. **`Seq Scan`** read every row; `Filter` kept the 69 hooks and removed the 10 brand rows.
2. **`Sort … top-N heapsort`** computed all 69 distances and kept only the best 5 (a small heap, 33 kB).
3. **`Limit`** returned 5 rows.
4. **`cost=21.31`** is the planner's estimate; **`actual time`** is the real one in milliseconds.

This is **exact** search: every distance computed.

Forcing the index (`SET LOCAL enable_seqscan = off`) gives an `Index Scan using kb_gemini_embedding_2_768_hnsw`,
**the same 5 rows in the same order**, and is slightly faster (0.16 vs 0.47 ms median). The planner estimates the
index at **164.59** before returning its first row, against **21.31** for the whole scan, so it never chooses the
index at this size.

Either way it takes under 1 ms, next to a 500–950 ms embedding request. **At 79 rows the index is decoration**;
it's there for when the library grows.

One more finding from the unfiltered query: its top 4 were the **brand sections** of the coffee-shop brand, ahead of
the POV pattern. Without `kind = 'hook'`, brand rules would be served as hook patterns. **The filter matters.**

### The filtering trap, caught for real
The coffee-shop query with `brand = 'samoviy-karvon'` (the *travel* brand, 5 rows, none of them near a coffee-shop
query), k = 5, forced through the index:

| ef_search | iterative_scan | Rows returned |
|---:|---|---:|
| 10 | off | **0** (`Rows Removed by Filter: 10`) |
| 40 (pgvector's default) | off | **3** |
| 10 | **strict_order** | **5** ✅ |
| 100 (ours) | **strict_order** | **5** ✅ |

- **No error, no warning: just fewer rows.** The index handed over its 10 (or 40) nearest candidates, none (or only
  3) were the travel brand's, and the filter threw the rest away.
- **Iterative scans** (pgvector ≥ 0.8) keep walking the graph until k rows pass the filter.
  - `strict_order` keeps the exact distance order, which we need for parity.
  - `relaxed_order` is faster but may return rows slightly out of order.
- **Do we need it at 79 rows?** Not today: the planner uses the exact scan. **We turned it on anyway**, because the
  day the table is big enough for the planner to choose the index is exactly the day brand filters would start
  silently returning 0–3 rules. A test with 300 rows forces the index and **fails without the setting**.
- **A lesson about testing:** the first filtered test used anor-va-dolchin's own brand query, and it didn't show the
  problem, because that brand's sections are its nearest rows. **A filter only bites when the rows it wants are far
  from the query.** Designing the test to hit that case was the point.

### The same on 20,000 rows (synthetic vectors, a throwaway table)
| Query | Plan | Rows | Time |
|---|---|---:|---:|
| k = 10, no filter | **Index Scan** (the planner chooses it now) | 10 | **0.40 ms** |
| the same, index disabled | Seq Scan + top-N heapsort | 10 (the same ids) | **87.4 ms** |
| k = 10, filter matching 10% of rows | Index Scan | **3** | 0.33 ms |
| the same + `iterative_scan = strict_order` | Index Scan | 10 | 0.69 ms |

| ef_search | 10 | 40 | 100 | 200 |
|---|---:|---:|---:|---:|
| recall@10 (vs exact) | 0.855 | 0.985 | 0.995 | 1.000 |

- **At 20,000 rows the index is ~200× faster**, and the planner picks it by itself.
- **`ef_search` trades recall for speed**, the same dial as the HNSW widget's `ef`.
- **The filtered recall was 0.395 without iterative scans, and 1.000 with them.** 0.395 ≈ the back-of-envelope
  "ef_search × the share that matches" = 40 × 10% = 4 rows out of 10. The optional exercise has you compute exactly
  that.
- **The index cost:** 78 MB for 20,000 rows, built in 11.5 s. For our 79 rows it's 328 kB.

### A detail about storage (TOAST)
Each vector is 3,076 bytes (768 × 4 + a small header). That's more than the ~2 kB Postgres keeps inside a row, so
every vector is stored **out of line** in a side table called TOAST. That's why the table is 152 kB but 1.2 MB with
TOAST. It's irrelevant at 79 rows, and a known tuning knob at millions (`SET STORAGE PLAIN`).

---

## 15. Errors that happened, and what they teach
| # | What happened | Lesson |
|---|---|---|
| E1 | `TypeError: 'Vector' object is not iterable` during the first migration: pgvector-python returns its own `Vector` class, not a numpy array | read the library's source when an assumption fails; the re-run was safe because upsert is idempotent |
| E2 | **the build printed the last byte of `.env`** (one character) to check for a trailing newline | a rule breach, owned in the notes; the right check prints only True/False. No variable was damaged |
| E3 | heredoc quoting and Git Bash rewriting `/var/lib/…` into a Windows path | tooling noise; `MSYS_NO_PATHCONV=1` |
| E4 | `'CallToolResult' object has no attribute 'isError'` in a scratch script | MCP SDK 2.x uses `is_error` in Python; `isError` is only the JSON name |
| E5 | the first `doctor` after installing psycopg took over 120 s | probably Windows scanning the new DLLs the first time; later runs took 4.8 s |
| E6 | a stopped database "times out" after 3 s instead of being "refused" | on Windows a closed localhost port isn't refused quickly; the message still names the fix |
| E7 | **the MCP server our client starts wouldn't have had `DATABASE_URL`** | `PASS_ENV` passes only named variables (Phase 4's least privilege), so `DATABASE_URL` and `VECTOR_STORE` were added. **The same applies to your Claude Desktop config** |
| E8 | the first version failed one call after every database restart | reconnect once to the same store; a mutation test without the retry fails |
| E9 | the filtering trap (§14) | fixed before it could ever happen, with a test that fails without the fix |

**"No silent fallback" is tested both ways:**
- the build temporarily changed `open_store()` to fall back to Chroma on a connection error, and **two tests caught
  it**;
- live, with the container stopped, `make --research` stopped in 6.8 s with
  `error: the pgvector database at 127.0.0.1:5433/shortsmith is unreachable` / `next: start Docker Desktop, then
  docker compose up -d db … or set VECTOR_STORE=chroma`. No run folder was made and no call was sent.
- Over MCP, `retrieve_hooks` returned an error result with the same next step, and 0 embedding requests.

**Honest limits:**
- **`doctor` now FAILs whenever Docker Desktop is closed**, even if you only want a video without `--research`.
  That's deliberate, but it's an inconvenience.
- A down database costs 3 s before the error.
- The numbers above 79 rows come from synthetic vectors.
- Chroma is still a dependency (for `VECTOR_STORE=chroma`, the parity check and the migration).
- Everything runs as the container's superuser, in the `public` schema, with no connection pool. That's fine for a
  local tool; part 16's list says what Phase 8–9 change.

---

## 16. Tests, and what Phase 8–9 will change

- **Tests:**
  - `uv run pytest` gives 427 passed and **22 skipped**, each skip with its reason ("DATABASE_URL is not set: start
    Docker Desktop, `docker compose up -d db`, then `uv run --env-file .env pytest`").
  - **With the database:** `uv run --env-file .env pytest` gives **449 passed** (+58).
  - **The same behaviour tests run on three stores:** an in-memory fake, real Chroma, and real pgvector in
    throwaway `test_…` tables that are dropped afterwards. That proves the interface really is all
    `KnowledgeBase` needs.
  - **Integration tests:** the column type and index definition, upsert and delete, score parity with Chroma on
    hand-made vectors, a selective filter through the index (with the plan checked), a lost connection reopened,
    a wrong password never shown in the error, and a copy from Chroma that needs no embedding.
- **Optional exercise:** `exercises/pgvector/pgvector_sql.py`, 19 tests:
  1. `similarity(distance, operator)` for `<=>`, `<->` and `<#>`;
  2. `topk_query(table, k, filters)`: build the parameterised SQL;
  3. `rows_left_after_filter(ef_search, k, matching)`: the trap's arithmetic.

  ```bash
  uv run pytest exercises/pgvector
  ```
- **What Phase 8–9 change when users and runs move into the same Postgres:**
  - **Schemas:** a `rag` schema for the knowledge base (it can be rebuilt any time) and an `app` schema for users
    and runs (they can't).
  - **Migrations:** versioned migrations (e.g. Alembic) for app tables, including `CREATE EXTENSION vector`.
  - **Roles:** a least-privilege role for the web app instead of the superuser.
  - **Connections:** a **connection pool** for FastAPI (`psycopg_pool`). Our `SET LOCAL` is already pool-safe.
  - **Backups:** scheduled `pg_dump`. Once users live there, `down -v` stops being an inconvenience and becomes
    a disaster.
  - **Deployment:** the same compose file on the Azure server, or Azure Database for PostgreSQL, which also offers
    pgvector.
  - **Possibly** the ledger and run history as tables, so "cost per user per day" becomes one SQL query.
- **One thing for you:** for the Claude Desktop test, the `env` block in its config needs `DATABASE_URL`, or start the
  server with `uv run shortsmith mcp`, which loads `.env` itself. Otherwise `retrieve_hooks` says "DATABASE_URL is
  not set".

---

## Check questions (optional, answer whenever you like)
1. What does `CREATE EXTENSION vector` add to Postgres?
2. Which operator gives cosine distance, and how do you turn it into the similarity Chroma reported?
3. Why must the HNSW index be built with `vector_cosine_ops` for our queries?
4. Why is `127.0.0.1:5433:5432` safer than `5433:5432`? Why 5433?
5. What does `docker compose down -v` delete, and why do we never run it?
6. Why `%s` placeholders and never f-strings in SQL?
7. With 79 rows, which plan do you expect from `EXPLAIN ANALYZE`, and why is that correct?
8. What is the post-filtering problem, and what do iterative scans do about it?
9. Why can we copy the vectors instead of re-embedding? When would we *have to* re-embed?
10. Why is a silent fallback to Chroma worse than a clear error?

**Our build**
11. Why one table per embedding model and size instead of a `collection` column?
12. What does `${SHORTSMITH_DB_PASSWORD:?…}` in the compose file protect you from?
13. How did the build prove the migration cost 0 embedding calls, not just report it?
14. In the real `EXPLAIN ANALYZE`, why did Postgres choose a sequential scan, and was the result exact?
15. The travel-brand query returned 0, then 3, then 5 rows. What changed each time?
16. Why turn on `hnsw.iterative_scan` when the planner doesn't even use the index at 79 rows?
17. Why does the MCP server need `DATABASE_URL` passed to it explicitly (E7)? What does that mean for Claude Desktop?
18. Why `SET LOCAL` and not `SET`?

## Interview angles
- "I moved RAG from Chroma to pgvector behind a small store interface: the same chunks, model and cosine metric,
  migrated without re-embedding, and proved identical top-k and identical hit@1/MRR before switching."
- "I can read the planner: at small sizes Postgres correctly prefers a sequential scan over the HNSW index; I know
  when the index pays off and how iterative scans fix filtered ANN queries."
- "Vectors live next to the application data, so joins, transactions and one backup cover everything; I'd move
  to a dedicated vector engine only when scale or filtered QPS demand it."
- "I reproduced the HNSW filtering trap on real data: a selective brand filter returned 0 of 5 rows through the index,
  and pgvector 0.8's iterative scans fixed it. On 20,000 rows, filtered recall went from 0.395 to 1.0."
- "Migration proof: 79 vectors copied bit-for-bit, a sync that would raise if it embedded anything, identical top-k and
  scores to 4 decimals, and hit@1 0.88 / MRR 0.914 through both stores."
