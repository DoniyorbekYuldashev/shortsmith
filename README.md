# Shortsmith

AI agents that turn one line of text into a finished ~30-second vertical short video:
research → script → storyboard → visuals + voice → edit → self-critique.

Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the plan and
[docs/LEARNING_LOG.md](docs/LEARNING_LOG.md) for what was learned at each step.

## Setup

```bash
uv sync                  # create .venv and install the exact versions from uv.lock
cp .env.example .env     # then fill in your keys, and a password of your own for the database
docker compose up -d db  # Postgres + pgvector for --research (start Docker Desktop first)
uv run shortsmith doctor # checks ffmpeg, which keys are set (never their values), media mode, the database
```

Requires: [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), and for `--research`
[Docker Desktop](https://www.docker.com/products/docker-desktop/) (or set `VECTOR_STORE=chroma`, below).

### The database (the RAG knowledge base)

`--research` and the MCP tool `retrieve_hooks` look up hook patterns and brand rules in Postgres with the
pgvector extension, running in Docker (`docker-compose.yml`: image `pgvector/pgvector:0.8.7-pg17`, on
`127.0.0.1:5433` only, data in the volume `shortsmith_pgdata`). Run these from the repo folder, with Docker
Desktop running:

```bash
docker compose up -d db      # start it (the first time: download the image, create the database)
docker compose ps            # is it up and "(healthy)"?
docker compose stop db       # stop it; the data stays
```

Never run `docker compose down -v`: `-v` deletes the volume, i.e. the stored vectors (`down` without `-v`
is safe). If the database is down, `--research` stops with one error that says what to do; nothing quietly
switches to another store. `VECTOR_STORE=chroma` uses the older local Chroma folder (`chroma/`) instead.

The first `--research` run embeds the 79 knowledge-base sections (79 free-tier embedding requests). If you
already have a Chroma index from before, copy it instead: 0 requests.

```bash
uv run --env-file .env python -m shortsmith.research.migrate                    # chroma/ -> Postgres
uv run --env-file .env python -m shortsmith.research.eval_retrieval --stores    # both stores give the same results?
```

## Usage

Run from the repo folder. `shortsmith --help` and `shortsmith <command> --help` explain everything.

```bash
uv run shortsmith make "coffee shop in Tashkent, cozy autumn vibe" --lang uz --research
uv run shortsmith list                        # runs, newest first, with status and cost
uv run shortsmith show latest                 # one run: script, critic rounds, cost per node, files
uv run shortsmith rerun-scene latest 3 --prompt "close-up of hands pouring tea"
uv run shortsmith cost --days 7               # spending by day, run, model; free-tier usage today
```

| command | what it does |
|---|---|
| `make TOPIC` | the multi-agent graph: research (optional), writer, director, hero-clip approval, producer, editor, critic loop. `--simple` runs the straight Phase 2 pipeline instead |
| `resume RUN` | continues a run after a crash, a budget stop or a paused approval; nothing finished is paid for again |
| `rerun-scene RUN N` | remakes one scene's image and/or voice (`--prompt`, `--narration`, `--image`, `--voice`) and re-cuts the video; replaced files go to `history/rerun_N/` |
| `list`, `show RUN`, `cost` | look at runs and spending; `--json` for scripts, `-o FILE` to write UTF-8 to a file |
| `doctor`, `mcp` | check the setup; start the MCP server (as Claude Desktop does) |

`RUN` is a run id, the start of one, or `latest`.

**Money.** Images and the hero clip are free placeholders unless you pass `--real`, which shows an estimate
and asks first (`--yes` for scripts). `--budget 0.05` is a hard limit per run in *list* price, checked before
every priced step; on the Gemini free tier every call is billed $0. A run that stops at its budget resumes with
`shortsmith resume RUN --budget <more>`. `--hero-clips 0` means Ken Burns only.

**Settings**, highest first: command-line flags > environment variables (`MOCK_MEDIA`, `SHORTSMITH_LANG`,
`SHORTSMITH_BUDGET`, `SHORTSMITH_CRITIC_MODEL`, `LLM_PROVIDER`, `VECTOR_STORE`, `DATABASE_URL`) > `.env` in the
current folder > defaults.

**Tests:** `uv run pytest` runs everything offline (the pgvector integration tests are skipped, with the
reason). With the database up, `uv run --env-file .env pytest` runs them too, in throwaway tables
(`-m pgvector` for only those).

**Exit codes:** 0 ok · 1 error · 2 usage · 3 paused for an approval · 4 over budget · 130 interrupted.

**Output.** Results go to stdout, progress to stderr. Piped or redirected output is plain (no colours or
spinners; tables as Markdown); `--plain` forces it, `NO_COLOR` turns colours off. Add `--debug` (anywhere) to
see a traceback instead of the one-line error. On Windows PowerShell 5.1, `>` and `|` re-read the output in the
console's code page and garble non-English text: use `-o FILE` (or `--json`, which is ASCII-only).

The Phase 2-6 module commands still work, e.g. `uv run python -m shortsmith.make "topic" --graph`, and
`uv run python -m shortsmith.mcp_server` for Claude Desktop.

## The web app (local)

A Next.js page that shows a run live: the pipeline timeline, scene thumbnails as they are made, the
hero-clip question as Yes/No buttons, critic rounds, tokens and list price per step, a budget stop with
"resume with a bigger budget", and the final video. Two terminals, both started from the repo folder:

```bash
uv run shortsmith serve      # terminal 1: the API on http://127.0.0.1:8010 (OpenAPI docs at /docs)
```

```bash
cd web
npm install                  # terminal 2, the first time only (Node 20.9+)
npm run dev                  # http://127.0.0.1:3000
```

- **Local only, mock only.** Both listen on 127.0.0.1; there is no login. Runs from the web always use the
  free placeholder media, whatever `MOCK_MEDIA` says (`--real` stays a CLI-only, confirmed choice). LLM calls go
  to the Gemini free tier as usual: the page shows list price and says that $0 is billed.
- **One run at a time.** Runs wait in a queue (at most 5); a run paused at the hero-clip question doesn't hold
  the queue. Research needs the database (`docker compose up -d db`); the home page says when it's down.
- **Stopping the server** (Ctrl+C) during a run keeps everything up to the last finished step; the page shows
  it as interrupted, with a Resume button. The CLI and the web share the same runs, ledger and checkpoints.
- Ports: 8010 (API; `shortsmith serve --port N`) and 3000 (web). `NEXT_PUBLIC_API_URL` points the page at
  another API address; `SHORTSMITH_WEB_ORIGINS` (comma-separated) changes which pages the API allows (CORS).

The page's TypeScript types are generated from the API's OpenAPI schema; after changing an API model, run
`npm run types` in `web/` (it rewrites `web/openapi.json` and `web/lib/api-schema.d.ts`; a test fails until you
do). Checks: `npm run build`, `npm run typecheck`, `npm test` (the event reducer, with Node's own test runner).
