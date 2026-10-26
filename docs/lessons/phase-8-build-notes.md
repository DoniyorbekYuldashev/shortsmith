# Phase 8 build notes (Steps 8.1-8.3: the web app)

Written by the build session for the teaching session. Everything below happened on this machine (Windows 11
Home with Smart App Control on, Windows PowerShell 5.1, Python 3.12, uv 0.13.1, **FastAPI 0.161.1**, **Starlette
1.9.0**, **uvicorn 0.54.0** (all three were already in the lock through mlflow and mcp: no new Python package),
Node 24.13.1, npm 11.8.0, **Next.js 16.4.3** (Turbopack), **React 19.3.1**, **TypeScript 5.9.3**,
**openapi-typescript 7.13.0**, Docker Desktop 29.4.0) on 2026-10-25 and 2026-10-26. Money spent: **$0**
(mock media only; Gemini free tier; see §8).

**Done-check, in one paragraph.** `uv run shortsmith serve` starts the API on 127.0.0.1:8010 (FastAPI + Server-
Sent Events; OpenAPI at `/docs`) and `npm run dev` in `web/` the Next.js page on 127.0.0.1:3000. Runs execute in a
worker thread, one at a time, behind a queue; every step of the graph becomes a typed event in
`runs/<id>/events.jsonl` that the page streams live and replays later. **Two full mock runs were watched in a
real browser** (the desktop app's browser pane): run 1 "night food stalls in the old town of Bukhara" (research,
hero clip "ask me", 1 revision) showed the timeline node by node with timers, tokens and list price, paused at
the hero-clip question with Yes/No buttons, queued behind run 2 after "Yes", hit an edge-tts DNS failure (shown
as the CLI's one-line message + next step + Resume), resumed from the page, and finished with **critic 6 -> revise
-> 6** drawn as a second "critic loop" row ($0.0245 list, 8 calls + 1 embedding). Run 2 "why Tashkent metro
stations look like palaces" ($0.01 budget, no revisions) showed thumbnails appearing one by one, a live ffmpeg
failure in the editor (E4), resumed from the page, then **stopped before the critic over budget** ("spent $0.0067
+ critic ~$0.0074 > budget $0.0100") and finished after **"Resume with a budget of $0.0200"** from the page (critic
5.8, $0.0132). A **reload mid-critic** rebuilt the page from the replay and kept the critic's timer counting; a
**finished run opened fresh** replays to "done · replayed"; the video seeks (206 Partial Content). **Phase total:
11 LLM calls + 1 embedding request, 22,415 tokens in, 8,326 out, $0.0377 list price, $0 billed** (3 of today's 20
Flash requests). **474 Python tests: 452 pass offline + 22 pgvector tests skipped without the database** (427 + 22
before; 24 new in test_api.py, the CLI help tests extended); `npm run build`, `npm run typecheck` and `npm test`
(9 reducer tests) pass; with the database up, `uv run --env-file .env pytest`: **474 passed**. `exercises/phase-8/` has 20 tests that fail with NotImplementedError by design.

**Three things the lesson should not miss**, all found by running it:
1. **An SSE event named `error` breaks EventSource** (E6): the browser dispatches a server message called `error`
   as its own error event, so `onerror` fired and every failed run's page said "reconnecting...". `open`, `error`
   and `message` are taken. Found only in a real browser; curl and the tests were happy.
2. **Read the worker's state before the checkpoint, never after** (E3): the other order reported a run that
   finished in between as "crashed". A check-then-act race, found by a test that failed 3 times in 4.
3. **A process-wide setting must be set where it's used, every time** (E2): `MOCK_MEDIA=1` set once at startup
   could be changed underneath; only a test that changed it mid-flight showed the web could have made paid
   images. Now every job sets it.

**Changes for the teaching session's files** (I did not edit them):
- **PLAN Phase 8**, an "as built" line: `shortsmith serve` (FastAPI + SSE on 127.0.0.1:8010) + Next.js 16 in `web/`
  (127.0.0.1:3000); one run at a time behind a queue; events persisted per run and replayed (`Last-Event-ID`);
  mock media only and local only; TypeScript types generated from the OpenAPI schema.
- **PLAN §4** (repo tree): `api.py` "FastAPI + SSE" becomes `api.py  jobs.py  events.py   # FastAPI + SSE; the
  worker and its queue; the event models and log`; under `web/`: `app/ components/ lib/ (api-schema.d.ts is
  generated) openapi.json`; `exercises/phase-8/`.
- **PLAN §3**: the "CLI / API" row could say FastAPI's built-in SSE (`fastapi.sse`, FastAPI >= 0.13x); "Frontend"
  Next.js 16 + TypeScript 5.9 (plain CSS, no Tailwind).
- **CLAUDE.md, Current progress:** Phase 8 built (commit below); the web app: `uv run shortsmith serve` (repo root)
  + `npm run dev` (in `web/`), http://127.0.0.1:3000; 474 tests (452 + 22 skipped offline); exercise waiting:
  `exercises/phase-8/` (optional: SSE wire format and replay).
- **CLAUDE.md, Gotchas** worth adding: (1) the web runs mock media only, whatever `MOCK_MEDIA` says; (2) after
  changing an API model, `npm run types` in `web/` (a test fails until then); (3) Next.js 16.4.3's native SWC
  binary loads under Smart App Control (no pin), and `agentRules: false` stops `next dev` writing an AGENTS.md;
  (4) never name an SSE event `open`, `error` or `message`; (5) **the database volume was gone on 2026-10-25
  (E11)**: if `docker compose up -d db` says `Volume ... Created`, the knowledge base is empty: restore it with
  `uv run --env-file .env python -m shortsmith.research.migrate` (from `chroma/`, 0 requests).
- The owner should know E11 (the volume) and E4 (a transient ffmpeg failure on a busy 7.8 GB machine).
- `docs/lessons/phase-8.md` exists (untracked, the teaching session's): not touched, not committed.
- Screenshots of the UI states are in the shared notes folder, `phase-8-screens/` (00 home ... 08 reload during
  the critic), not in the repo.

---

## 1. What was built

### Files

| File | Status | What it is |
|---|---|---|
| `src/shortsmith/api.py` | new | the FastAPI app (`create_app()`): 13 endpoints, CORS, file serving, the SSE stream, `serve()` (uvicorn on 127.0.0.1), `python -m shortsmith.api --openapi FILE` |
| `src/shortsmith/jobs.py` | new | the background work: `Job` (= `runs/<id>/job.json`), `Worker` (one thread + a FIFO), `Emitter` (graph hooks -> events), `Jobs` (submit, `live()` status, `recover()` after a restart, `_drive()`) |
| `src/shortsmith/events.py` | new | 13 event models (a discriminated union on `type`), `EventLog` (`runs/<id>/events.jsonl`, ids 1, 2, 3, ... per run) |
| `src/shortsmith/agents/producer.py` | changed | `ON_MADE` hook (a context variable, like graph's `ON_START`): told after each image, voice and clip |
| `src/shortsmith/agents/graph.py` | changed | `drive(on_made=...)` sets it; nothing else |
| `src/shortsmith/budget.py` | changed | `suggested()` moved here from `cli._suggest_budget` (the CLI's `next:` line and the page's button share it) |
| `src/shortsmith/runs.py` | changed | `RunStatus.kind` gains `"queued"` (only the API says it); `job.json` counts as a run file (a queued run is listed) |
| `src/shortsmith/cli.py` | changed | `shortsmith serve [--port N] [--no-trace]`; `short_notes()`: a failed tool's stderr is shortened from the middle, not cut to its end (E4) |
| `src/shortsmith/console.py` | changed | a colour for "queued" |
| `tests/test_api.py` | new | 24 tests (§7) |
| `tests/test_cli.py` | changed | `serve` in the help tests |
| `exercises/phase-8/` | new | optional: `sse_wire.py` (write SSE, parse SSE, replay after Last-Event-ID) + 20 tests |
| `exercises/phase-7/budget_rules.py` | changed | one docstring line: `suggest_budget` now mirrors `budget.suggested` |
| `web/` | new | the Next.js app (below) |
| `.gitignore` | changed | `web/out/`, `web/next-env.d.ts`, `*.tsbuildinfo` (node_modules/ and .next/ were already there) |
| `README.md` | changed | "The web app (local)" section |

### The endpoints (`uv run shortsmith serve` -> http://127.0.0.1:8010, docs at `/docs`)

| method + path | answers | what it does |
|---|---|---|
| `POST /runs` | 202 `Accepted` | queue a new run (`StartRun`: topic, language, research, brand, approve ask/yes/no, max_rounds 0-5, pass_score, budget_usd, hero_clips 0/1). Checks first, before any folder exists: an unknown brand (400), a full queue (429), the CLI's `preflight` (key, ffmpeg, and the vector store if research: 503 with message + hint) |
| `GET /runs?limit=50` | `list[RunSummary]` | the runs, newest first: `runs.list_runs()` with the worker's status on top |
| `GET /runs/{id}` | `RunDetail` | `runs.detail()` (= `shortsmith show --json`), same overlay |
| `GET /runs/{id}/events` | `text/event-stream` / 204 | the run's events: everything after `?after=` or the `Last-Event-ID` header, then new ones live; a heartbeat comment every 15 s of quiet; ends after `done`; `follow=false` ends after the replay; 204 = finished and nothing left to send (EventSource stops reconnecting) |
| `GET /runs/{id}/log?after=0` | `list[Event]` | the same events as one JSON list (curl, tests, and the TypeScript types: §1 "types") |
| `POST /runs/{id}/approval` | 202 | `{"answer": "yes"\|"no"}` for a run paused at the hero-clip question (409 if it isn't) |
| `POST /runs/{id}/resume` | 202 | `{"budget_usd": 0.02}` optional. After a budget stop, a failed node or a server restart. 409: finished, paused (answer instead), already queued/running, a CLI process driving it, made with `--real` (CLI only), no checkpoint (unless it was queued once and never started: then it starts again) |
| `GET /runs/{id}/files/{path}` | the file | png, jpg, mp4, mp3, srt from inside the run folder only; Range requests (206) |
| `GET /costs?days=&run=` | `CostReport` | `costs.report()` (= `shortsmith cost --json`) |
| `GET /queue` | `QueueView` | running id + its node, the waiting ids, the limit (5) |
| `GET /options` | `Options` | languages, brands (data/hooks/brands/), the form's defaults |
| `GET /health` | `Health` | provider + model + critic model, key set (true/false only), media "mock", free tier, ffmpeg, research ready + `StoreStatus` (URL masked), problems as message + hint, the queue |
| `GET /docs`, `/openapi.json` | | FastAPI's own |

Every refusal has one shape: `{"detail": {"message": "...", "hint": "..."}}` (the CLI's `error:` + `next:`).
FastAPI's validation errors (422) keep FastAPI's shape; the page turns both into one line.

### The events (`events.py`; `type` is the SSE `event:` name)

| type | when | carries |
|---|---|---|
| `queued` | a job enters the queue | position, the run ids ahead |
| `run_started` | the worker takes the job (new run or a resume) | resumed, next nodes, the settings, budget, `spent_usd` before this job, provider/model/critic model, `free_tier` |
| `node_started` | graph `ON_START` (after the budget check passed) | node, round, what it's doing, estimate (list price) + its basis |
| `node_finished` | `drive`'s `on_update` | seconds, calls, prompt/completion tokens, list price, `graph.describe()` (the CLI's done line) |
| `media_made` | producer `ON_MADE` | image/voice/clip, scene, round, path, url `?v=<round>` |
| `approval_needed` | the interrupt's approver | the `HeroQuestion` + `waiting` (true = paused for a person; false = the run's policy answers) |
| `approval_answered` | the policy, or `POST /approval` | yes/no, by person/policy |
| `video_made` | the editor finished | round, url |
| `critic_report` | the critic finished | score, pass score, summary, weak scenes (reason, fix), frames + their urls, model, `then` revise/done (`graph.route_after_critic`), reason (`graph.stop_reason`) |
| `revision` | the reviser finished | each scene: before/after narration and picture, `changed` |
| `budget_stopped` | `BudgetExceeded` | node, spent, estimate, limit, `suggested_usd` (`budget.suggested`), message |
| `failed` | a node raised, or the server stopped mid-run (`interrupted`) | node, the CLI's message + hint (`cli.explain`) |
| `done` | the graph reached END | final url, scores per round, stop reason, seconds, calls/tokens/list/billed from the ledger, budget |

Every event also has `id` (1, 2, 3, ... per run) and `at` (UTC). The union is a Pydantic discriminated union
(`Field(discriminator="type")`), so FastAPI documents it as `oneOf` + a discriminator and parsing a line picks
the right class by its `type`.

### The job model

```
POST /runs ──> checks ──> graph.new_state() (the folder) ──> runs/<id>/job.json {kind: start, state, status: queued}
                                                        └──> Worker.submit(job) ──> event `queued`
Worker thread (one): take the oldest job ──> job.json status running ──> MOCK_MEDIA=1
   ──> graph.open_graph() + runs.marked_running(folder)   (the same `.running` marker as the CLI)
   ──> event `run_started` ──> graph.drive(app, payload, run_id, approve, on_update, on_start, on_made)
   ──> one of: done | budget_stopped | failed | (paused: approval_needed was already sent) ──> job.json settled
```

- **One run at a time**, FIFO, at most 5 waiting (429 beyond). A run paused at the question **doesn't hold
  the worker**: its job ends; the answer queues a resume job (`Command(resume=answer)`).
- **The same pipeline**: `graph.drive()` with the node wrapper's `ON_START` hook and the new `ON_MADE`;
  `BudgetExceeded` from the node wrapper's `budget.check()`; `cli.explain()` for the error text;
  `budget.suggested()` for the button. Nothing is forked.
- **Status**: `runs.status_of()` reads the checkpoint, which can't see this process's own worker (the
  `.running` marker holds the server's pid, and `running_pid()` never counts its own process), so the API
  asks the worker first (`Jobs.live()`) and the checkpoint second (E3).
- **Restart**: `Jobs.recover()` at startup: a job left "running" (and its stale marker) -> a `failed` event
  with `interrupted: true` ("the server stopped while this run was at critic") and the job settled; jobs left
  "queued" are queued again, oldest first. Only folders with a `job.json` (web runs) are touched; a run a live
  CLI process drives is left alone.

### The frontend (`web/`)

```
web/
  package.json            exact versions; scripts: dev, build, start (all on 127.0.0.1:3000), typecheck, test, types
  tsconfig.json           strict; allowImportingTsExtensions + erasableSyntaxOnly (Node can run the .ts tests)
  next.config.ts          reactStrictMode; agentRules: false (E5)
  openapi.json            GENERATED by `npm run types` (python -m shortsmith.api --openapi)
  app/layout.tsx          the header (New video, Runs; "local only · mock media · billed $0 on the Gemini free tier")
  app/globals.css         one stylesheet: tokens, light/dark, the timeline, scene cards, badges
  app/page.tsx            Home: the form, Setup (from /health), Recent runs
  app/runs/page.tsx       the runs table (refreshes every 5 s)
  app/runs/[id]/page.tsx  -> components/RunView.tsx
  components/RunView.tsx  EventSource -> useReducer(reduce); refetches GET /runs/{id} after script-changing nodes
  components/Timeline.tsx one row per round (first pass, then each critic loop); the running node pulses with a timer
  components/Scenes.tsx   6 cards; a thumbnail appears on its media_made event; voice / hero clip / revised tags
  components/Critic.tsx   score vs pass mark, the edge's decision, weak scenes (reason -> fix), the frames, the revisions
  components/Actions.tsx  ApprovalCard (Yes/No), BudgetCard (resume with $X), ErrorCard (message + next + Resume)
  components/Costs.tsx    per node: calls, tokens in/out, list price, time; total; "Billed: $0"; the budget meter
  components/StatusBadge.tsx
  lib/api-schema.d.ts     GENERATED by openapi-typescript (never edited by hand)
  lib/types.ts            short names: RunSummary, RunDetail, ..., RunEvent (the union), EventOf<"node_started">
  lib/api.ts              one typed function per endpoint; errors -> {message, hint}
  lib/run-state.ts        the pure reducer: events -> page state; rounds(), byNode(), spent()
  lib/run-state.test.ts   9 tests with node:test (`npm test`)
  lib/format.ts           usd() like the CLI's, tokens, seconds, labels
```

### How the types are shared

`npm run types` (in `web/`) = `uv run python -m shortsmith.api --openapi web/openapi.json` (no server needed:
`create_app().openapi()`) + `openapi-typescript openapi.json -o lib/api-schema.d.ts`. `lib/types.ts` gives the
generated types short names. The event union reaches the schema through `GET /runs/{id}/log` (`list[Event]`):
the SSE endpoint itself can't type its items here (§5 D5). Two checks keep them in sync:
`tests/test_api.py::test_the_events_are_typed_in_the_openapi_schema_...` fails when `web/openapi.json` differs
from what the API generates now ("run `npm run types`"), and then `npm run build` fails on any page that uses
a field that changed. The reducer's `switch` ends in `const unknown: never = event`, and `EVENT_TYPES` has a
compile-time completeness check: a new event type in Python is a TypeScript error until the page handles it.

### Commands and ports

| what | where | command |
|---|---|---|
| the API | terminal 1, repo root | `uv run shortsmith serve` (127.0.0.1:8010; `--port N`, `--no-trace`) |
| the page | terminal 2, `web/` | `npm install` (first time), then `npm run dev` (127.0.0.1:3000) |
| research | repo root, Docker Desktop running | `docker compose up -d db` |
| types after an API change | `web/` | `npm run types` |
| checks | `web/` | `npm run build`, `npm run typecheck`, `npm test` |
| Python tests | repo root | `uv run pytest` (`--env-file .env` adds the 22 pgvector tests) |
| the exercise | repo root | `uv run pytest exercises/phase-8` |

Ports: 8010 (8000 is Yukdosh's; 5432 the Windows Postgres; 5433 ours), 3000 (Next's default). Both bound to
127.0.0.1, not 0.0.0.0. `NEXT_PUBLIC_API_URL` and `SHORTSMITH_WEB_ORIGINS` change the pairing.

---

## 2. Real outputs

### The two live runs, as their events tell it (UTC)

**Run 1, `2026-10-25_144556`**: "night food stalls in the old town of Bukhara", en, research, hero clip "ask me",
up to 1 revision -> "Bukhara Night Walk".

| time | event | |
|---|---|---|
| 10-25 09:45:56 | queued | position 1 |
| 09:46:11 | run_started | 15 s later: the worker's first job imports MLflow first (tracing.setup) |
| 09:46:12-09:46:32 | researcher 8.4 s (1,370 in / 334 out, $0.0012), writer 6.2 s ($0.0053, 2 calls), director 5.7 s ($0.0018) | |
| 09:46:32 | approval_needed, waiting | the run's job ends; the worker is free |
| 09:47:25 | approval_answered "yes" by person -> queued | **behind run 2** (`ahead: [2026-10-25_144704]`) |
| 09:47:52 | run_started (resumed before approve_hero) | 0.1 s after run 2's job ended |
| ... 09:48:17 | producer: images and voices of scenes 1-6 as `media_made` events | |
| 09:57:11 | failed at producer: `ClientConnectorDNSError ... speech.platform.bing.com:443 ... [getaddrinfo failed]` | E10 |
| 10-26 12:37:18 | queued (Resume pressed on the page) | behind run 2's critic |
| 12:43:00 | run_started (resumed before producer) | 5 min 42 s in the queue |
| 12:43:00-12:43:30 | producer 2.8 s (only what was missing), editor 27.2 s, video_made | |
| 12:43:30-12:49:29 | critic **359 s** (Google's free queue): score 6, weak [1] "a generic walking-tour cliché" -> revise | |
| 12:49:43 | revise 13.7 s: scene 1's narration rewritten | approve_hero: no new clip needed |
| 12:50:12 | producer (scene 1's voice), editor 23.6 s, video_made `?v=1` | |
| 12:52:32 | critic 140 s: score 6, weak [1] -> **max rounds**; done | 8 calls + 1 embedding, 20,443 tokens, $0.0245, video 28.1 s |

**Run 2, `2026-10-25_144704`**: "why Tashkent metro stations look like palaces", en, hero clip "yes", no
revisions, budget $0.0100 -> "Palaces Underground".

| time | event | |
|---|---|---|
| 10-25 09:47:04 | queued, run_started (30 ms apart: the worker was free; MLflow already loaded) | |
| 09:47:04-09:47:16 | writer 7.3 s ($0.0054), director 5.0 s ($0.0013) | |
| 09:47:16 | approval_needed + approval_answered "yes" by policy | |
| 09:47:16-09:47:51 | producer 34.7 s: 13 `media_made` events (6 images, 6 voices, the clip) | |
| 09:47:52 | **failed at editor**: ffmpeg/libx264 "Generic error in an external library" | E4 |
| 10-26 12:35:07 | queued (Resume), run_started 10 s later (MLflow again: a restarted server) | |
| 12:35:53 | editor 35.7 s, video_made; **budget_stopped before critic**: spent $0.0067 + ~$0.0074 > $0.0100, suggested $0.02 | |
| 12:36:47 | queued + run_started (Resume with $0.0200, written into the checkpoint) | |
| 12:43:00 | critic **373 s**: 5.8, weak [1, 4]; done (0 revisions allowed) | 4 calls, 10,298 tokens, $0.0132, video 34.3 s |

### A raw SSE stream (curl, run 2 during its critic, reconnecting with `Last-Event-ID: 35`)

```
> curl -N http://127.0.0.1:8010/runs/2026-10-25_144704/events -H "Last-Event-ID: 35"
: run 2026-10-25_144704
retry: 3000

event: run_started
data: {"id":36,"at":"2026-10-26T12:36:47.488445Z","type":"run_started","resumed":true,"next":["critic"],"topic":"why Tashkent metro stations look like palaces","language":"en","research":false,"brand":null,"max_rounds":0, ...}
id: 36

event: node_started
data: {"id":37,"at":"2026-10-26T12:36:47.497709Z","type":"node_started","node":"critic","round":0,"doing":"watching 4 frames and judging the video","estimate_usd":0.007401,"basis":"critic $0.0074 (most expensive of the last 10)"}
id: 37

: keep-alive
```
(the comment and `retry` first; only the events after 35; the heartbeat 15 s after the last event, while the
critic waited in Google's queue). Headers: `content-type: text/event-stream; charset=utf-8`, `x-accel-buffering: no`,
`access-control-allow-origin: http://127.0.0.1:3000` when the page asks. After `done` with nothing new:
`Last-Event-ID: 40` -> **204**.

### Other answers (curl, Docker stopped)

```
GET /health  (4.3 s: `docker info` + a 3 s database connect timeout)
{"ok":true,"version":"0.1.0","provider":"gemini","model":"gemini-3.5-flash-lite","critic_model":"gemini-3.8-flash",
 "key_set":true,"youtube_key_set":true,"media":"mock","free_tier":true,"ffmpeg":true,"research_ready":false,
 "vector_store":{"backend":"pgvector","table":"kb_gemini_embedding_2_768","url":"postgresql://shortsmith:***@127.0.0.1:5433/shortsmith",
   "docker":"the engine is not running (start Docker Desktop)","docker_running":false,"reachable":false,
   "error":"can't connect to the pgvector database at 127.0.0.1:5433/shortsmith (connection timeout expired)", ...},
 "problems":[{"message":"research is off: can't connect to ...","hint":"start Docker Desktop, then `docker compose up -d db` ..."}],
 "queue":{"running":null,"node":null,"queued":[],"max_queued":5}}

POST /runs {"topic":"tea","research":true}       -> 503 {"detail":{"message":"can't connect to the pgvector database at 127.0.0.1:5433/shortsmith (connection timeout expired)","hint":"start Docker Desktop, then `docker compose up -d db` (in the repo folder), or set VECTOR_STORE=chroma"}}
POST /runs {"topic":"tea","media":"real"}        -> 422 [{"type":"extra_forbidden","loc":["body","media"],"msg":"Extra inputs are not permitted"}]
POST /runs/2026-10-25_144704/approval (finished) -> 409 {"detail":{"message":"run 2026-10-25_144704 is not waiting for an answer (finished)","hint":""}}
GET /runs/2026-10-25_144704/files/..%2F..%2F.env -> 404 {"detail":{"message":"no file '../../.env' in run 2026-10-25_144704","hint":""}}
GET .../files/final.mp4  Range: bytes=0-1023     -> 206, content-range: bytes 0-1023/4358943, accept-ranges: bytes, etag, cache-control: no-cache
```

### The UI states (seen in the browser; screenshots in the notes folder's `phase-8-screens/`)

- **Home** (00): the form (topic, language, brand, hero clip ask/yes/no, critic rounds, budget, research), "Setup"
  from `/health` (LLM, key set, mock media, "Gemini free tier: $0 billed", research ready with 79 rows, the queue),
  recent runs with status badges and list price. Research is greyed out with the hint when the database is down.
- **Timeline mid-run** (01): Researcher done (8.4 s, 1,370 in · 334 out, $0.0012), Writer pulsing "running 1.2 s",
  the rest "waiting"; under it "Writer: writing 3 script options, then picking one · est. $0.0063 list price".
- **The question** (02): an amber card "Hero clip for scene 2?", the director's motion, "Mock media: a free
  placeholder clip now. With real media this would cost ~$0.30", buttons "Yes, make the clip" / "No, use Ken
  Burns"; the Approval node amber "waiting for your answer".
- **Queued** (03): "Queued: one run at a time (the free tier is slow and Flash allows 20 requests a day). Waiting
  for 2026-10-25_144704" (a link).
- **Thumbnails** (04): placeholder cards appearing one by one during the producer, each with its motion and words.
- **Error** (05): a red card "Error at editor: ffmpeg failed ...", "Next: the run is saved after its last finished
  node: shortsmith resume ...", a "Resume from the last finished step" button; the Editor node red.
- **Budget stop** (06): "Stopped: over budget before the critic", the numbers, an input with 0.02 and "Resume with a
  budget of $0.0200 · $0.0200 surely finishes (every round still allowed)"; the budget meter "$0.0067 of $0.0100".
- **Replay** (07): "done · replayed", every node green with time, tokens and price (Critic "6 min 13 s, 5,315 in ·
  651 out, $0.0064"), the critic round with its weak scenes and fixes, the video.
- **Reload during the critic** (08): the whole page back in under a second; "Critic: ... 28.2 s" counting on from
  the event's own start time.
- **Run 1 at the end** (text): rows "first pass" (7 nodes) and "critic loop 1" (Revise, Approval, Producer, Editor,
  Critic); Critic panel 6 -> "revising scene(s) 1" with the old and new narration, then 6 -> "max rounds (1
  revisions; last score 6)"; the cost table: Critic ×2 2 calls 10,891 in / 1,664 out $0.0144 8 min 19 s, total 8
  calls, 20,443 tokens, $0.0245, "Billed: $0 (Gemini free tier)"; tags "revised in loop 1", "hero clip".
- **Runs list**: run 1 said "running (critic)" while it ran (the worker's overlay), then finished.
- **The video**: `<video>` loads with 206 responses, `duration 34.33`, `seekable 0-34.33`, seeking to 25 s works,
  1080x1920.

### Timings that matter
- `uv run shortsmith serve` -> answering in ~2 s; the first job then waits 10-15 s for MLflow's import.
- `npm run dev` ready in 0.8 s; the first page compile ~1.2 s; `npm run build` ~10 s (compile 4.4 s, TypeScript 2.5 s).
- Free-tier critic calls: **359, 373 and 140 s** (Phase 7: 135-266 s). Writer/director/researcher 5-8 s each.
- Editor 24-36 s, producer 35 s for a full pass (edge-tts), 3 s for one voice.
- The page receives an event within ~0.3 s of the worker writing it (the stream's poll interval).

---

## 3. Errors hit, how they were found, how they were fixed

**E1. `TypeError: 'EventSourceResponse' object is not iterable`** (500 on the first SSE test). FastAPI 0.161
has SSE built in (`fastapi.sse`: `EventSourceResponse`, `ServerSentEvent`, `format_sse_event`, a keep-alive
ping), but `response_class=EventSourceResponse` on a route means "this function *yields* events": FastAPI then
iterates whatever the function returns. Ours returns either a 204 or a stream, so it can't be a generator.
Found by `test_a_run_streams_its_events_...`; the traceback ended in `fastapi/routing.py ... iterate_in_threadpool`.
Fix: no `response_class`; the endpoint returns `EventSourceResponse(stream())` itself (a StreamingResponse with
the right media type), the 200 content type is documented by hand, and the bytes come from FastAPI's own
`format_sse_event`. Lesson: check what the installed version does before trusting a tutorial (older ones use
the sse-starlette package; FastAPI now has its own, with different rules).

**E2. A real bug the tests caught: the web could have made PAID images.** `test_media_is_always_mock_from_the_web`
sets `MOCK_MEDIA=0` after the server started (as a `.env` with `MOCK_MEDIA=0` would before a later job) and
expects placeholders. It failed: `assert '0' == '1'`. The lifespan set `MOCK_MEDIA=1` once at startup, but
`images.mock_media()` reads the environment at the moment an image is made, so anything that changes it later
wins. The run's own state said `media: "mock"` the whole time, which is why only the test noticed. Fix:
`Jobs._run()` sets `MOCK_MEDIA=1` at the start of every job, next to the comment saying why.

**E3. A run that was running showed "crashed at critic"** (a flaky test: 3 of 4 runs failed). The status
overlay read the checkpoint first (`next = critic`, no error: "crashed") and asked the worker second; when the
run finished in between, the worker said "not mine" and the checkpoint's stale answer stood. Found with a
debug test that printed the worker's state and the API's answer at the same moment (the worker said running,
the API said finished/crashed). A check-then-act race (TOCTOU). Fix: `Jobs.live(run_id, view)` takes a
snapshot of the worker FIRST, then the checkpoint is read; reading in that order, a run that finishes in
between is reported as running (true when we looked), never as crashed. The test also asserted on every
critic start instead of the first one: fixed too.

**E4. The editor failed live: `ffmpeg failed with exit code 3752568763 ... [vost#0:0/libx264 @ ...] Terminating
thread with return code -542398533 (Generic error in an external library)`.** Run 2 on 2026-10-25, 0.85 s into
the editor, after scene 1's clip; scene 2's Ken Burns render failed. 3752568763 is -542398533 as an unsigned
32-bit number = FFmpeg's `AVERROR_EXTERNAL` (an error inside libx264). Two findings:
(a) the message was useless: `cli.explain()` kept the LAST 300 characters of ffmpeg's stderr, and ffmpeg 8+
prints the cause first and its "Task finished ... Terminating thread" lines last; the cause was cut off. Fix:
`cli.short_notes()` keeps the first two thirds and the last third, and the server log now gets the whole stderr
(`jobs.py`). Test: `test_a_long_tool_error_keeps_its_first_words_...`.
(b) the cause: NOT reproduced. The same edit on a copy of the folder, run alone, worked (46.4 s, a 34.3 s video),
and the resume from the page the next day cut the video fine. At the time the machine (7.8 GB RAM; 1.6 GB free
even later, with fewer things open) had Docker's VM just started, the Next dev server, the API with MLflow, and
the browser; a 2160x3840 zoompan into x264 needs a lot of memory, and x264 failing to allocate is the most
likely reading. Honest status: unproven; the next time, the log will show ffmpeg's first line.

**E5. Next.js 16.4's `next dev` wrote a file into `web/`:** `✓ Generated AGENTS.md for AI agents. Set
agentRules: false in next.config to disable.` The file (679 bytes) tells AI coding agents to read the Next docs
in node_modules first. Not ours, and this phase creates no extra docs: `agentRules: false` in
`next.config.ts` (the option exists in Next's config types), file deleted; a restart didn't recreate it.

**E6. Every page of a failed run said "reconnecting..." although the stream was fine.** Found live (the
connection label next to the status badge). curl showed a healthy stream; the browser's network log showed one
200; a hand-made `EventSource` in the page's console logged `["error", readyState 1]` exactly when the run's
`event: error` message arrived. **EventSource dispatches a server message named `error` as its own `error`
event**, so `onerror` fired for our message. `open`, `error` and `message` are names the browser already uses.
Fix: the type is `failed` (Python, TypeScript, tests); a test asserts no event type is `open`, `error` or
`message`; the two live runs' event files were rewritten (`"type":"error"` -> `"failed"`, one line each).

**E7. `shortsmith serve --port 8011` -> `No such option: --port`.** Phase 7's E3 again: with `metavar="PORT"`
(the upper-cased parameter name), Typer named the option itself `--PORT` (`--help` showed `--PORT PORT`). Found
by `test_serve_listens_on_localhost_only`. Fix: `typer.Option("--port", metavar="NUMBER")`, with a comment.

**E8. `tsc`: `Property 'pass_score' is missing ... required in type StartRun`.** openapi-typescript 7 types a
field that has a `default` as always present (`--default-non-nullable`, on by default): right for responses
(the API always sends it), too strict for request bodies. Tried `--default-non-nullable false`: then every
defaulted field of every response became optional (`r.calls` possibly undefined in `Costs.tsx`: 5 errors).
Kept the default and made the form send every field, `pass_score` included (it's what the form does anyway).

**E9. A test race:** `/queue` said `node: None` because the worker hadn't reached the writer's `on_start` yet
when the test asked. The test now polls until it has.

**E10. edge-tts lost DNS mid-run (live):** `ClientConnectorDNSError: Cannot connect to host
speech.platform.bing.com:443 ... [getaddrinfo failed]`, run 1's producer, 9 minutes after its last voice
(09:48:17 -> 09:57:11 UTC, with tts.py's own retries in between). The page showed it as the CLI's one-line error
+ next step + a Resume button; resumed from the page the next day: only the missing voices were made.

**E11. Docker (not code).** The first `Start-Process "Docker Desktop.exe"` didn't stay up (no Docker Desktop
process 4 minutes later; `docker info` calls hung); the second start answered in ~35 s (engine 29.4.0), Dashboard
window only, no sign-in or terms dialog. **`docker compose up -d db` said `Volume shortsmith_pgdata Created`**:
the volume, and the old `shortsmith-db-1` container, were gone; the image was still there (which fits a
`docker compose down -v`, or a reset of Docker's data) sometime between the pgvector build and this session.
This session didn't do it. Restored for free: `python -m shortsmith.research.migrate` copied the 79 rows from
`chroma/` (hashes match, max vector difference 0, 0 embedding requests, the ledger stayed at 198 lines).

**E12 (noise, not fixed).** The API's log shows `Exception in callback _ProactorBasePipeTransport.
_call_connection_lost ... ConnectionResetError: [WinError 10054]` when the browser drops a connection (the
`<video>` element aborts range reads it no longer needs: the browser's network log shows them as
`206 ... ERR_ABORTED`). Python's asyncio on Windows (the Proactor loop) logs it; harmless. A logging filter
could hide it; left visible and documented.

**E14. Two pages were invisible to git.** `.gitignore`'s `runs/` (the pipeline's output folder, Phase 2) also
matches `web/app/runs/`, so `web/app/runs/page.tsx` and `web/app/runs/[id]/page.tsx` were silently ignored: the
commit would have shipped a web app without its runs pages. Found by reading `git status --untracked-files=all`
before committing. Fix: `/runs/` (anchored to the repo root); `git check-ignore` confirms `runs/ledger.jsonl` and
the run folders are still ignored.

**E15 (not ours, flaky).** One full run with the database had 1 failure: `test_rag.py::test_cosine_ignores_how_
long_the_vectors_are[chroma]` -> `chromadb.errors.InternalError: ... Error creating hnsw segment reader: Nothing
found on disk` (Chroma's Rust index in a temp folder). The file alone passed 3 of 3 times and the next full run
passed 474 of 474. A Phase 5 Chroma test; noted, not chased.

**E13 (the build tool's own, for completeness).** Driving the page by screenshot coordinates typed the topic
twice once (a stale screenshot) and a `next.config.ts` edit restarted the dev server, which reloaded the page and
emptied the form; the live checks then used element refs and read the form's values back with JavaScript.

---

## 4. Design decisions and why

**D1. Server-Sent Events, not WebSockets or polling.** The data flows one way (server -> page); the browser's
`EventSource` reconnects by itself and sends `Last-Event-ID`, so resuming a stream is a header, not a protocol
we'd write; it's plain HTTP (curl can read it, CORS works as for any GET, proxies pass it), and the page's two
actions (answer, resume) are ordinary POSTs. WebSockets earn their extra machinery (a second protocol, our own
reconnect and replay, ping/pong) when both sides talk a lot: chat, games, collaborative editing. Polling
(GET /runs/{id}/log every N seconds) would also work and is simpler still, but it's late by up to N seconds and
asks even when nothing happens; SSE pushes the event the moment the worker writes it (well, within 0.3 s: D3).

**D2. FastAPI's own SSE writer, our own endpoint.** `fastapi.sse.format_sse_event` writes the bytes (one tested
implementation, the same order the spec uses); the endpoint returns the response itself so it can also answer
**204 No Content**: the SSE way to tell a browser "this run is finished and you have everything: stop
reconnecting". `retry: 3000` comes first; a comment line `: keep-alive` goes out after 15 quiet seconds (a free
critic call is minutes of silence, and proxies cut idle connections; EventSource ignores comments).

**D3. Events are a file per run; memory only knows each run's last id.** `runs/<id>/events.jsonl`, append-only,
one line per event, ids 1, 2, 3, ... per run (not global: a run's file is all you need to replay it). Replay
for a reconnect = the lines with a larger id; a page opened tomorrow = all of them; a server restart loses
nothing. The stream wakes every 0.3 s and compares two integers (its last sent id, the log's last id); it reads
the file only when they differ. A half-written last line (a crash mid-write) is not an event until it ends with
a newline. Alternative: an in-memory pub/sub (an asyncio queue per subscriber, filled from the worker thread with
`call_soon_threadsafe`): lower latency, but it needs the replay-then-subscribe dance without gaps or duplicates,
and it lives in one process. With several API processes (Phase 9) the file becomes a table or Redis (Postgres
LISTEN/NOTIFY or Redis pub/sub to wake the streams).

**D4. One worker thread and a FIFO** (jobs.py's docstring has the comparison). BackgroundTasks: no queue, no
status, nothing after a restart. A thread pool with one worker is what we have, plus the order we show. A real
queue (Celery, RQ, Arq + Redis, separate worker processes) once a job must survive the web process dying, run on
another machine, retry by itself, or once several web processes take requests. Why one: the free tier allows a
few requests a minute and 20 Flash requests a day; two runs at once would only slow both and hit 429s. A paused
run (waiting for the hero-clip answer) ends its job, so it never blocks the queue. The thread is a daemon:
stopping the server doesn't wait 10 minutes for a run; its checkpoint has every finished node and the next start
marks it interrupted (E-test: `test_a_restart_marks_the_running_run_interrupted_and_requeues_the_waiting_one`).
Cost of the thread choice: a crash that kills the process (a segfault in a native library, `os._exit`) takes the
API down with it; a worker *process* would isolate that.

**D5. Types: generated from OpenAPI, not written twice.** The Pydantic models are the one source; FastAPI writes
the schema; openapi-typescript turns it into TypeScript (`npm run types`). A hand-written copy drifts silently;
this one fails a Python test until regenerated, then fails the TypeScript build wherever a page uses a changed
field. The event union is documented through `GET /runs/{id}/log` (`list[Event]`): FastAPI documents SSE item
types only for `yield` endpoints (D2's 204 rules that out), and the log endpoint is useful anyway. No typed fetch
library (openapi-fetch): `lib/api.ts` is eight small functions using the generated types.

**D6. No Tailwind, no component library: one CSS file.** The page is small (three screens); plain CSS with a few
variables gives light and dark mode and is readable without learning a second language. It also avoids two
native binaries (Tailwind 4's `@tailwindcss/oxide` and `lightningcss`), each another chance for Smart App Control
to block a fresh download. Next's own native SWC binary (110 MB, 2 days old) loaded fine (§2).

**D7. The page calls the API directly, with CORS** (allowed origins: http://127.0.0.1:3000 and
http://localhost:3000; methods GET and POST; headers Content-Type and Last-Event-ID). A Next.js rewrite would
avoid CORS, but a dev-server proxy can buffer a streaming response, and then nothing is live. `<img>` and
`<video>` don't need CORS at all.

**D8. Local only and mock only, now.** No login exists yet, so the API listens on 127.0.0.1 only: nothing on the
network can start runs. Mock media is forced per job (E2): a web page that can spend money needs a login, a spend
cap and rate limits first (Phase 9). `--real` stays in the CLI, where it shows an estimate and asks.

**D9. The status the page shows** = the worker's word (running, queued) if it has one, else the checkpoint's
(`runs.status_of`), read in that order (E3). `"queued"` became a `RunStatus` kind in runs.py; the CLI never says
it. A queued run has only `job.json`, which now counts as a run file, so `shortsmith list` lists it too (as "no
checkpoint" until it starts).

**D10. The page's state is a pure reducer over the events** (`lib/run-state.ts`): a live run, a reload mid-run (the
stream replays from event 1) and a run opened tomorrow all go through the same function, so they can't disagree.
An event whose id the page already has is ignored (a reconnect that overlaps). Tested with Node's built-in test
runner: Node 24 runs `.ts` files by stripping the types, so no Vitest/Jest (more packages, and their native
build tools: esbuild, rollup); `erasableSyntaxOnly` in tsconfig keeps the code strippable (no enums, no parameter
properties).

**D11. The hero-clip policy is per run**: `ask` (pause; the page shows Yes/No), `yes`, `no`. It's kept in
`job.json`, so the resume after an answer uses it again if a revision changes the hero picture and asks again.

**D12. Files: only media types, only inside the run folder, `?v=<round>` URLs.** The path is resolved and must
stay inside `runs/<id>/` (`..`, `%2e%2e`, backslashes, absolute paths: all 404, tested); `job.json`,
`events.jsonl` and the marker are never served. A remade file keeps its name, so each event's URL carries the
round (`?v=1`) and the response says `Cache-Control: no-cache` (the browser asks again; the ETag makes it a 304).
Starlette's `FileResponse` answers Range requests (206), so the video seeks.

**D13. Exact versions.** `package.json` pins exact versions (`npm install --save-exact`) and `package-lock.json`
is committed; TypeScript 5.9.3, not 7.0 (the 7.x line is the Go rewrite with a different API; Next's build uses
the JavaScript `typescript` package).

---

## 5. What Phase 9 must change for a public demo

- **Who may start a run**: a login or at least an API key per user; rate limits per user and per IP (a run is
  ~6-11 LLM calls and the free tier's 20 Flash requests a day are shared); a **global spend cap** checked before
  every run and every priced node (the per-run `--budget` already exists in the graph; a daily total doesn't).
- **Real media from the web**, if ever: only behind the login + the cap, with the estimate shown and confirmed
  in the page, as the CLI does.
- **CORS**: the real frontend origin (the Vercel domain) instead of localhost; **HTTPS** everywhere (Vercel does
  the page; the API behind Azure's TLS); bind 0.0.0.0 only behind that proxy.
- **Where runs live**: the API's disk is not shared or durable on a scaled-out host. Runs' media go to blob
  storage (Azure Blob) with signed URLs; events, jobs, the ledger and checkpoints go to Postgres (LangGraph has
  a Postgres checkpointer; the knowledge base is already there). SSE streams then wake on Postgres LISTEN/NOTIFY
  or Redis pub/sub instead of a file's last id.
- **A real job queue** (Arq or RQ on Redis, or Celery): workers as separate processes (or containers) that
  survive API restarts, with retries and timeouts; the API only enqueues and reads.
- **Hosting**: Vercel for `web/` (`NEXT_PUBLIC_API_URL` = the API's HTTPS address); Azure App Service or a small
  VM for the API + worker, ffmpeg installed; secrets in the host's settings, never in the repo.
- **Abuse and privacy**: topic length and content checks, no personal data in prompts, logs without topics if
  they are personal.

---

## 6. Honest limits

- **One process.** The worker is a thread inside the API; a hard crash of the process stops both (the run is
  resumable). Two API processes would each run their own worker and could drive the same run twice.
- **Polling inside the stream**: up to 0.3 s late, and every open page wakes 3 times a second (a dict lookup).
  Fine for one person; not for hundreds of open pages.
- **The editor's x264 failure (E4) is not explained**, only made explainable next time.
- **The timeline shows the researcher only after the run starts** (before `run_started` the page doesn't know the
  run's settings), and a run the CLI made has no events: its page shows the checkpoint's numbers (scenes, critic,
  cost per node, video) without a timeline.
- **No auth, no rate limit, no CSRF protection**: acceptable only because it listens on 127.0.0.1.
- **The browser checks were done by an automated browser pane**, not by a person clicking; the page wasn't
  checked on a phone-sized screen beyond the CSS's one breakpoint.
- **Windows-only checks**: the paths, ports and Smart App Control results are this machine's.
- **The `.running` marker uses a process id**: a reused pid after a reboot could look alive (as in Phase 7).

---

## 7. Tests

**Python** (`tests/test_api.py`, 24 tests, offline, ~20 s): FastAPI's `TestClient` (its lifespan starts the real
worker thread) against `test_graph.py`'s `world` (the fake LLM answering by schema, fake voice, ffmpeg, frames
and clips; a real LangGraph + SQLite checkpointer in a temp folder). `env` adds a key, a fake `which` for
ffmpeg, a fake ffprobe and a healthy store, like test_cli's.
- a whole run: the SSE body starts with the comment + `retry: 3000`; ids 1..N with no gaps; `event:` = the
  data's `type`; the order queued, run_started, writer, director, approve_hero (question, policy answer, the
  node again), producer (image 1, voice 1, image 2, ... the clip), editor + video_made, critic (5 -> revise, weak
  [3]), revise, ..., critic (8 -> done), done; the writer's numbers (2 calls, 200/100 tokens); frame URLs; the
  revision's before/after; `done` = the ledger (6 calls, the same list price, $0 billed);
- the wire format byte for byte = `fastapi.sse.format_sse_event`;
- `approve: ask` -> paused with the question (`~$0.30`), resume refused (409: answer instead), the answer -> the
  rest of the run, the director not called again, a second answer refused; `no` -> Ken Burns for the hero scene;
- the budget stop before the critic with the suggested $0.04, `list` says "over budget before critic", resume with
  $0.05 finishes, the budget saved in the checkpoint, the writer not paid again, a finished run can't resume;
- a failing node = the CLI's message + `shortsmith resume <id>` hint, "crashed at director", resume finishes;
- `Last-Event-ID: 5` -> 6..N; `?after=5` the same; the log endpoint's `after`; everything seen + finished -> 204;
- a stream opened mid-run (the critic held on a gate): it replays, the API says "running (critic)", then the rest
  arrives live on the same connection;
- `follow=false` ends after the replay even while the run waits for an answer;
- one run at a time (the writer held on a gate): the second is queued with position 2 and the first ahead, a third
  gets 429 (`max_queued=1`), `/queue` shows the running node, the queued run is listed with its topic, its
  `run_started` comes after the first's `done`;
- files: png/jpg served with their types; 10 attempts refused with 404 and no content (`..%2F..%2Fsecret.txt`,
  `%2e%2e/%2e%2e/.env`, `..%5C..%5C`, an absolute path, a run id with `..`, `.env` as a run id, job.json,
  events.jsonl, media.json);
- Range: `bytes=2-5` -> 206 with exactly those bytes and `content-range`; past the end -> 416; `accept-ranges`;
- list/show/costs: `RunSummary.model_validate(row) == runs.list_runs()[0]` (the CLI's model), detail sections,
  costs total and `?run=`, a 404's `{message, hint}`, `/options`;
- refused before anything starts: a blank topic, 9 rounds, 2 hero clips, `"media": "real"` (422: extra fields
  forbidden), an unknown brand (400), a missing key (503, the CLI's message); no LLM call, no run folder;
- mock always (E2) and a `--real` run's resume refused with the CLI command as the hint;
- health with the database down (the REAL `stores.status`, DATABASE_URL at 127.0.0.1:1): 200, `ok`, research not
  ready, the masked URL, the hint `docker compose up -d db`; research refused with 503 before a folder exists;
- no secret in any response (/health, /runs, a run, its log and events, /costs, /options, /queue, /openapi.json)
  with fake keys and a DB password set;
- CORS: the two local origins allowed, another one not;
- a restart: a run "killed" before its critic (job running, a stale marker with a dead pid, the last event the
  critic starting) gets a `failed` + `interrupted` event, the marker removed, "crashed at critic", resume finishes;
  a queued job with no checkpoint is queued again and finishes; one writer call per run in total;
- the event log: ids per run, a half-written line ignored by a fresh reader, `after()`;
- the schema: the event union in `/runs/{id}/log`, `id` required, no event named open/error/message, and
  `web/openapi.json` equal to what the API generates now;
- `serve`: uvicorn gets host 127.0.0.1 and the port, the access log on stderr, nothing on stdout;
- a long tool error keeps its first words (E4).

`tests/test_cli.py`: `serve` in the command tree and its help (`--port`, `--no-trace`).

**TypeScript** (`npm test` in web/: `node --test lib/run-state.test.ts`, 9 tests, ~0.2 s): a first pass folds into
steps, scenes, the video and a critic round; approve_hero starting twice is one step; a critic that sends scenes
back opens the next round's row; the loop round and `done` (the ledger's total wins); a waiting question pauses,
the answer and the resume go on; a budget stop, then a resume clears it; an error marks the open step failed; a
replayed event changes nothing; a CLI-started run counts what it spent before. `npm run typecheck` (tsc) and
`npm run build` pass.

**The exercise** (`uv run pytest exercises/phase-8`): 20 tests fail with NotImplementedError by design; the same
20 pass against a reference solution (checked in a scratch folder, not committed).

---

## 8. What it cost

From `runs/ledger.jsonl`, the lines tagged with this phase's two runs (list prices; **billed $0**):

| run | step | model | calls | in | out (incl. thinking) | list price |
|---|---|---|---:|---:|---:|---:|
| 1 (Bukhara) | rag (embedding) | gemini-embedding-2 | 1 | 18 | 0 | $0.000004 |
| 1 | researcher | gemini-3.5-flash-lite | 1 | 1,370 | 334 | $0.001246 |
| 1 | writer + writer-pick | gemini-3.5-flash-lite | 2 | 1,430 | 1,939 | $0.005277 |
| 1 | director | gemini-3.5-flash-lite | 1 | 658 | 649 | $0.001820 |
| 1 | critic (2 rounds) | gemini-3.8-flash | 2 | 10,891 | 1,664 | $0.014409 |
| 1 | revise | gemini-3.5-flash-lite | 1 | 877 | 613 | $0.001796 |
| 2 (metro) | writer + writer-pick | gemini-3.5-flash-lite | 2 | 1,163 | 2,023 | $0.005407 |
| 2 | director | gemini-3.5-flash-lite | 1 | 693 | 453 | $0.001340 |
| 2 | critic | gemini-3.8-flash | 1 | 5,315 | 651 | $0.006428 |
| **total** | | | **12** (11 LLM + 1 embedding) | **22,415** | **8,326** | **$0.037724** |

- gemini-3.8-flash: 3 requests on 2026-10-26 (of 20 a day). The plan's ceiling was ~30 calls.
- YouTube: run 1's research, 201 units (a search, Phase 5's `broader()` second search after too few Shorts, one
  videos call) of our 2,000 cap. `broader()` again dropped the LAST word: "night food stalls in the old town of
  Bukhara" -> "night food stalls in the old town of" (the Phase 7 open item, seen again).
- Media $0: placeholders, two mock hero clips, edge-tts, ffmpeg. The knowledge-base restore: 0 requests.
- Tests and the exercise: fakes only.

---

## 9. Open items

- **E4**: if the editor's x264 failure comes back, the log now shows ffmpeg's first lines; on this 7.8 GB machine,
  stopping Docker when research isn't needed leaves more room for ffmpeg.
- **E11**: the database volume was recreated; it holds the 79 rows again. Whoever deleted it should know `down -v`.
- The approval label "you said yes: goes on when the worker is free" (a fix after the live check) was checked by
  the type checker and the build, not in a new live run (it would cost another writer + director).
- `done`'s video link now uses the round of the last cut (`?v=1`, not `?v=2`); the live runs were made before the
  fix (their `done` events say `?v=2` / `?v=1`); harmless (cache-busting only).
- The YouTube `broader()` query still drops the place name (Phase 7's open item).
- The critic loop still doesn't converge (run 1: 6 -> 6, a new complaint each round), as in Phases 6-7.
- Phase 9 (§5): auth, rate limits, a spend cap, HTTPS, CORS for the real origin, runs in blob storage + Postgres,
  a real job queue, Vercel + Azure.
- The Windows asyncio `ConnectionResetError` log noise (E12) could get a logging filter.
