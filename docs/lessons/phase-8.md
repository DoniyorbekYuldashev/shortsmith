# Phase 8 — The web app: watching an AI pipeline work, live, in the browser

> Read at your own pace, ask anything. Parts 0–12 are the concepts (the code there is illustrative). Parts 13–17
> are what was really built and run (from `docs/lessons/phase-8-build-notes.md`, commit 7f6f822): the API, the
> events, the UI, real streams, errors, tests.

---

## 0. The problem: the best part is invisible

A run takes 1–10 minutes. In the terminal you see one line per node. But the interesting things are visual:
- the scene pictures appearing one by one;
- the Critic looking at frames and sending a scene back;
- the question "this clip costs $0.30, continue?";
- the final video.

Phase 8 puts all of it in a browser, **live**. Two programs work together:
```
browser (Next.js app, port 3000)  ──HTTP──►  API (FastAPI, port 8010)  ──►  the Phase 6 graph (in a background worker)
        ▲                                         │
        └────── a stream of events (SSE) ◄────────┘
```
- **The backend** is FastAPI in Python. It starts runs, answers questions, serves files, and **streams events**.
- **The frontend** is Next.js in TypeScript. It shows the timeline, the thumbnails, the buttons and the video.

---

## 1. HTTP in five minutes (the language between the two)

The browser and the API talk in **HTTP**. Each exchange is a request and a response:
```
POST /runs HTTP/1.1                         HTTP/1.1 201 Created
Content-Type: application/json              Content-Type: application/json
                                            
{"topic": "plov secrets", "lang": "uz"}     {"run_id": "2026-10-26_101500", "status": "queued"}
```
- **Method:** what you want to do.
  - `GET` reads (it must not change anything);
  - `POST` creates or does something;
  - `PUT`/`PATCH` change something; `DELETE` removes it.
- **Path:** what you're talking about. `/runs`, `/runs/2026-10-26_101500`, `/runs/…/events`.
- **Status code:** how it went.

  | Code | Meaning |
  |---|---|
  | 200 | ok |
  | 201 | created |
  | 202 | accepted, will happen later (perfect for "run started") |
  | 206 | partial content (a piece of a video, §7) |
  | 400 / 422 | your request is wrong (422 = failed validation) |
  | 404 | not found |
  | 409 | conflict (e.g. "this run is already running") |
  | 500 | the server broke |

- **Headers:** extra information: `Content-Type`, `Last-Event-ID`, `Range`, `Origin`.
- **Body:** usually JSON.

**REST** is a style for designing these: **nouns in the path, verbs as methods**. `POST /runs` starts a run;
`GET /runs/{id}` reads one; `POST /runs/{id}/approval` answers its question.

---

## 2. FastAPI: an API from type hints (again)

FastAPI does for web APIs what Typer did for the CLI and Pydantic did for LLM output: **the types are the
contract**.
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Shortsmith")

class StartRun(BaseModel):
    topic: str
    language: Literal["en", "uz", "ru"] = "en"
    research: bool = False

@app.post("/runs", status_code=202)
def start_run(req: StartRun) -> RunSummary:      # the request body is validated, the response is shaped
    ...

@app.get("/runs/{run_id}")
def show_run(run_id: str) -> RunDetail:           # Phase 7's RunDetail model, reused as is
    ...
```
- **The request body is validated by Pydantic.** A missing topic or a language `"de"` is rejected with
  **422** before your code runs.
- **The return type shapes the response.** Fields not in the model are never sent, which protects you from
  leaking internal data.
- **OpenAPI for free:** FastAPI writes a machine-readable description of every endpoint and model
  (`/openapi.json`), plus an interactive page at **`/docs`** where you can try every endpoint in the browser.
- **That's why Phase 7 split `runs.py`/`costs.py` from the terminal code:** the same `RunSummary`, `RunDetail` and
  `CostReport` serve the CLI's `--json` and the API, with no duplicated logic.

`def` vs `async def` endpoints: FastAPI runs a plain `def` in a **thread pool** and an `async def` on the
**event loop** (Phase 4's async lesson). Blocking work (our graph, ffmpeg, a slow LLM call) must never run
directly inside an `async def`, or it freezes every other request.

---

## 3. Long jobs: don't make the browser wait

A run takes minutes; an HTTP request should take milliseconds to seconds. So:
1. `POST /runs` **puts the job in a queue and returns at once**: `202 Accepted` with the run id.
2. A **background worker** takes jobs from the queue and runs the graph.
3. The browser **watches** the job's progress through a separate channel (§4).

**Where the worker can live:**

| Option | How | Good for | Weak spot |
|---|---|---|---|
| FastAPI `BackgroundTasks` | runs after the response, in the same process | small, short tasks (send an email) | no queue, no limit, lost if the server restarts |
| **a worker thread + a queue in the API process** | one thread takes jobs one by one | **one machine, one user, slow jobs** (us now) | lost queue on restart; one process only |
| a job-queue system (Celery, RQ, Arq) + Redis | separate worker processes read jobs from Redis | many users, many workers, retries, scheduling | more services to run |

**Why one run at a time for us:** the free tier is slow, and each Flash model allows 20 requests a day. Two
parallel runs would only fight for the same quota. A queue makes the rest wait their turn, visibly.

**What survives a restart:** the queue in memory doesn't, but **the run does**. Its state is checkpointed after
every node (Phase 6), so a run that was "running" when the server died shows as crashed at a node and can be
**resumed**. The same `.running` marker idea as the CLI (Phase 7).

---

## 4. Getting live updates: polling vs SSE vs WebSockets

| | How | Direction | Good for |
|---|---|---|---|
| **Polling** | the browser asks `GET /runs/{id}` every 2 seconds | browser → server, repeatedly | simplest; wasteful; up to 2 s late |
| **SSE** (Server-Sent Events) | one long HTTP response that the server **keeps writing to** | **server → browser only** | progress, notifications, LLM token streaming |
| **WebSockets** | a two-way connection after an HTTP "upgrade" | both ways | chat, games, collaborative editing |

**We use SSE**, because the updates flow one way (server → browser), and the few things the browser sends
(start, approve, resume) are ordinary `POST` requests. SSE is **plain HTTP**: it works through proxies, needs no
special server, and **the browser reconnects by itself**. (LLM chat apps stream their answers token by token with
the same technique.)

**What an SSE stream looks like on the wire:** text, one event per block, a blank line between events:
```
id: 7
event: node_finished
data: {"node": "writer", "seconds": 13.1, "prompt_tokens": 2355, "completion_tokens": 3402, "cost_usd": 0.0092}

id: 8
event: node_started
data: {"node": "director"}

: heartbeat

```
- `id:` numbers the events. `event:` is the type. `data:` is the payload (we send JSON). The blank line ends the event.
- A line starting with `:` is a **comment**. A **heartbeat** comment every ~15 s stops proxies from closing a
  quiet connection (the Critic can be silent for 5 minutes on the free tier).
- The response has `Content-Type: text/event-stream` and simply never ends until the run does.

**In the browser**, three lines:
```ts
const es = new EventSource(`http://127.0.0.1:8010/runs/${runId}/events`);
es.addEventListener("node_finished", (e) => update(JSON.parse(e.data)));
es.addEventListener("done", () => es.close());
```

---

## 5. Reconnect and replay: events as the source of truth

**Reconnect.** If the connection drops (a reload, Wi-Fi blip, laptop sleep), `EventSource` reconnects
automatically and sends the header **`Last-Event-ID: 8`**. A good server then sends **only the events after 8**.
For that, the server must **remember** events. We save each one to `runs/<id>/events.jsonl` with an increasing id.

**Replay.** Because every event is saved, a run page opened **tomorrow** can replay the whole run: the same
stream, read from the file, ending with `done`.

**Event sourcing.** This is a well-known pattern:
- the **list of events is the truth**;
- the screen is **computed from it**, by folding the events one by one into a state.

```ts
function reduce(state: RunView, ev: RunEvent): RunView {
  switch (ev.type) {
    case "node_started":  return { ...state, current: ev.node };
    case "node_finished": return { ...state, nodes: { ...state.nodes, [ev.node]: ev } };
    case "image_made":    return { ...state, thumbnails: [...state.thumbnails, ev.url] };
    case "critic":        return { ...state, rounds: [...state.rounds, ev] };
    case "approval":      return { ...state, question: ev };
    case "done":          return { ...state, finished: true, video: ev.video };
  }
}
```
It's the same idea as LangGraph's reducers (Phase 6), on the browser side. A live run and a replayed one go
through the **same** function, so they can't look different.

---

## 6. CORS: why the browser blocks the API by default

The page comes from `http://localhost:3000`, and the API is `http://127.0.0.1:8010`: a **different origin**
(scheme + host + port). Browsers apply the **same-origin policy**: JavaScript on one origin may not read responses
from another, **unless that server says it's allowed**.
- The server answers with `Access-Control-Allow-Origin: http://localhost:3000`.
- For "non-simple" requests (a `POST` with JSON), the browser first sends an **OPTIONS "preflight"** request to ask.
- FastAPI's `CORSMiddleware` handles both.

**Allow exactly your frontend, never `*`** on an API that changes things. CORS protects **users' browsers**, not
your server: `curl` ignores it completely, so it's no substitute for authentication (Phase 9).

---

## 7. Serving files: video seeking and path traversal

**Range requests.** When you drag the video's progress bar, the browser doesn't download the whole file. It asks
for a piece: `Range: bytes=1048576-`. The server answers **`206 Partial Content`** with just those bytes. Without
Range support, a video may play but **can't be seeked**, and some browsers refuse to play it at all.

**Path traversal.** An endpoint like `GET /runs/{id}/files/{path}` must never serve
`/runs/x/files/../../.env`. The rule from Phase 4's MCP server, again:
- check the run id against a strict pattern;
- **resolve** the final path;
- refuse anything that isn't **inside** that run's folder.

The build includes a test that tries `../`.

---

## 8. The frontend: JavaScript, TypeScript, React, Next.js

**JavaScript** is the language browsers run. **TypeScript** is JavaScript **with types**, checked before the code
runs and then removed. It's the same idea as Python type hints, except TypeScript's compiler actually refuses to
build when types don't fit:
```ts
type NodeName = "researcher" | "writer" | "director" | "approve_hero" | "producer" | "editor" | "critic" | "revise";

interface NodeFinished {
  type: "node_finished";
  node: NodeName;
  seconds: number;
  cost_usd: number;
}

type RunEvent = NodeFinished | NodeStarted | ImageMade | CriticRound | ApprovalAsked | Done;   // a "union"

function label(ev: RunEvent): string {
  if (ev.type === "node_finished") return `${ev.node} ${ev.seconds} s`;   // TypeScript now KNOWS ev has .seconds
  return ev.type;
}
```
- **Union types + checking a field** ("narrowing") are how TypeScript models our event types: the same idea as
  the `Literal` and discriminated models in Pydantic.

**React** builds a UI from **components**, functions that return what to show:
```tsx
function NodeChip({ name, seconds }: { name: string; seconds?: number }) {
  return <span className="chip">{name}{seconds !== undefined && ` · ${seconds.toFixed(1)} s`}</span>;
}
```
- **Props:** inputs from the parent (`name`, `seconds`).
- **State:** data the component remembers (`useState`, `useReducer`). **When state changes, React redraws**, so you
  never update the screen by hand: you update the state, and the screen follows.
- **Effects** (`useEffect`): code that runs after drawing. For example, opening the `EventSource` when the run
  page appears, and closing it when you leave.
- That angle-bracket syntax is **JSX/TSX**: HTML-like markup inside TypeScript.

**Next.js** is a framework around React:
- **routing by folders:** `app/page.tsx` is `/`, `app/runs/[id]/page.tsx` is `/runs/<any id>`;
- **server and client components:** server components render on the server; anything interactive (state,
  `EventSource`, buttons) is a **client component**, marked with `"use client"`;
- **a dev server with hot reload** (`npm run dev`): save a file and the page updates;
- **a production build** (`npm run build`) that type-checks and optimises everything. It's the "tests" of the
  frontend: if it doesn't build, it doesn't ship.

**npm** is Python's pip + uv for JavaScript: `package.json` ≈ `pyproject.toml`, `package-lock.json` ≈ `uv.lock`,
`node_modules/` ≈ `.venv`.

---

## 9. Keeping both sides in sync: types from OpenAPI

The backend's Pydantic models and the frontend's TypeScript types describe **the same data**. If they're written
twice by hand, they **drift**: rename a field in Python, and the UI breaks at runtime.

**Generate the TypeScript types from the API's OpenAPI schema** (e.g. with `openapi-typescript`): one source of
truth, the Pydantic models. Change a model, regenerate, and the TypeScript compiler shows every place in the UI
that must change. It's the same "single source of truth" idea as the writer's rules and the validator sharing
constants (Phase 1), now across two languages.

---

## 10. Local-only and mock-only, on purpose

- **The API binds to 127.0.0.1:** only this PC can reach it. There's **no login**, so it must not be reachable
  from the network.
- **No real media from the web in this phase:** a button that spends money needs authentication, rate limits and
  a spend cap first. That's Phase 9's work for the public demo.
- **Secrets never leave the server:** the API returns masked URLs and set/not-set flags, as `doctor` does.
- **Same rules as the CLI:** a budget stops cleanly; the database being down is a clear status, not a fallback.

---

## 11. What "live" really requires (a checklist)

1. A job runs **outside** the request.
2. Its progress becomes **typed events**.
3. Events are **saved** with ids.
4. The server **streams** them (SSE) and **resumes** from `Last-Event-ID`.
5. The browser **folds** them into a view state (a reducer).
6. Questions (approval) go back as **normal POSTs**, and the run continues from its **checkpoint**.
7. Files are served **safely** (no traversal) and **seekably** (Range).

Every piece has a reason, and every piece is testable without a browser.

---

## 12. The shape of the result (before the real thing)

```
http://localhost:3000                   topic: [ coffee shop in Tashkent          ]  [uz ▾] [✓ research]  [ Start ]

http://localhost:3000/runs/<id>         researcher ✓ 4.3s $0.0012 → writer ✓ 13.1s $0.0092 → director ✓ →
                                        approve_hero ?  [ Make the clip (~$0.30 real; free mock) ]  [ No, Ken Burns ]
                                        ▢ ▢ ▢ ▢ ▢ ▢   ← thumbnails appear here
                                        critic: round 0 · 5.5 · weak [1] → revise → critic: round 1 · 6.2 …
                                        ▶ final.mp4                    total $0.0355 list · $0 billed
```
The real screens come from the build; part 14 replaces this sketch.

---

## 13. Our API and events

**Versions:** FastAPI 0.161.1, Starlette 1.9.0 and uvicorn 0.54.0 were all already installed (MLflow and MCP use
them), so there was no new Python package. On the frontend: Next.js 16.4.3, React 19.3.1, TypeScript 5.9.3 and
openapi-typescript 7.13.0. Next's native SWC binary loaded fine under Smart App Control.

**How to run it** (two terminals):
```bash
uv run shortsmith serve
```
```bash
npm run dev
```
- The first command runs at the repo root: the API on http://127.0.0.1:8010, with docs at `/docs`.
- The second runs in the `web/` folder (`npm install` once, before the first time): the page on
  http://127.0.0.1:3000.
- For research, also start Docker Desktop and run `docker compose up -d db`.

### The new Python files
| File | What it is |
|---|---|
| `api.py` | the FastAPI app: 13 endpoints, CORS, file serving, the SSE stream, `serve()` |
| `jobs.py` | the background work: `Job` (= `runs/<id>/job.json`), `Worker` (one thread + a queue), `Emitter` (graph hooks → events), `recover()` after a restart |
| `events.py` | 13 event models (a union keyed on `type`) and `EventLog` (`runs/<id>/events.jsonl`) |

**The pipeline wasn't forked.** The only changes in the graph:
- an `ON_MADE` hook in the producer, which tells the worker after each image, voice and clip;
- `drive(on_made=…)`.

The node wrapper's `ON_START` hook (Phase 7) and `on_update` already existed.

### The endpoints
| Method + path | Answer | What it does |
|---|---|---|
| `POST /runs` | **202** | queue a run (topic, language, research, brand, hero clip ask/yes/no, rounds, pass score, budget). Checked **before** any folder exists: unknown brand → 400, full queue → **429**, database down with research → **503** + hint |
| `GET /runs` | `list[RunSummary]` | Phase 7's `runs.list_runs()` + the worker's live status |
| `GET /runs/{id}` | `RunDetail` | = `shortsmith show --json` |
| `GET /runs/{id}/events` | `text/event-stream` / **204** | the live stream; resumes after `Last-Event-ID`; heartbeat every 15 s; **204 = finished and nothing left** |
| `GET /runs/{id}/log` | `list[Event]` | the same events as one JSON list |
| `POST /runs/{id}/approval` | 202 / **409** | `{"answer": "yes"\|"no"}`; 409 if the run isn't waiting |
| `POST /runs/{id}/resume` | 202 / 409 | after a budget stop, a failure or a restart; optional `{"budget_usd": 0.02}` |
| `GET /runs/{id}/files/{path}` | the file, **206** for ranges | images, video, audio, subtitles; inside the run folder only |
| `GET /costs` | `CostReport` | = `shortsmith cost --json` |
| `GET /queue`, `/options`, `/health` | | the queue; the form's choices; setup status (keys as true/false, the URL masked) |

**Every refusal has the CLI's shape:** `{"detail": {"message": "...", "hint": "..."}}`, i.e. `error:` + `next:`.

### The 13 event types (each also has `id` 1, 2, 3, … per run and `at`)
`queued` · `run_started` · `node_started` · `node_finished` · `media_made` · `approval_needed` ·
`approval_answered` · `video_made` · `critic_report` · `revision` · `budget_stopped` · `failed` · `done`

- They're a **Pydantic discriminated union** (`Field(discriminator="type")`): reading a line picks the right
  class from its `type`. That's the Python side of TypeScript's union narrowing in §8.
- **`failed`, not `error`.** Bug E6 below explains why that name matters.

### The job model
```
POST /runs ─► checks ─► new run folder ─► runs/<id>/job.json {status: queued} ─► Worker.submit ─► event "queued"
Worker thread (one): oldest job ─► status running ─► MOCK_MEDIA=1 ─► graph.drive(...) with the hooks
   ─► ends in one of: done | budget_stopped | failed | paused (approval_needed was already sent)
```
- **One run at a time, at most 5 waiting** (a 6th gets 429).
- **A paused run doesn't hold the worker.** Its job ends at the question, and your answer queues a new "resume"
  job (`Command(resume=answer)`). In the live test, run 1 answered "yes" and then waited behind run 2.
- **After a server restart:** a run left "running" gets a `failed` event saying "the server stopped while this run
  was at critic", and you resume it. Runs left "queued" are queued again.

### How the types are shared
`npm run types` (in `web/`) writes `openapi.json` **from the Python app without starting a server**, then
`openapi-typescript` turns it into `lib/api-schema.d.ts`. That gives two safety nets:
1. a **Python test fails** if `web/openapi.json` is older than the API ("run `npm run types`");
2. then **`npm run build` fails** on every page that uses a changed field.

The page's reducer ends with `const unknown: never = event`: a **new event type in Python is a TypeScript compile
error** until the page handles it.

---

## 14. The UI, live: what you see

**Home:**
- the form: topic, language, brand, hero clip (ask/yes/no), Critic rounds, budget, research;
- a **Setup** box from `/health`: LLM, key set, "mock media", "Gemini free tier: $0 billed", research ready (79
  rows), the queue;
- your recent runs with status badges and cost.

**When the database is down**, "research" is greyed out with the hint.

**The run page, in order:**
- **Timeline:** one row per round ("first pass", then "critic loop 1", …).
  - The running node **pulses with a timer**.
  - Finished nodes show seconds, tokens in/out and list price: `Researcher 8.4 s · 1,370 in · 334 out · $0.0012`.
  - Under it: `Writer: writing 3 script options, then picking one · est. $0.0063 list price`.
- **The hero-clip question:** an amber card, "Hero clip for scene 2?", with the director's motion, "Mock media: a
  free placeholder clip now. With real media this would cost ~$0.30", and the buttons **Yes, make the clip** /
  **No, use Ken Burns**.
- **Scenes:** six cards. **Each thumbnail appears the moment its `media_made` event arrives**, with tags such as
  "voice", "hero clip" and "revised in loop 1".
- **Critic:** the score against the pass mark, the edge's decision ("revising scene 1" or "max rounds"), the weak
  scenes with reason → fix, the frames it saw, and the revisions with the old and new narration.
- **Costs:** per node (calls, tokens, list price, time), the total, **"Billed: $0 (Gemini free tier)"**, and a
  budget meter.
- **Cards for problems:**
  - **error:** the CLI's one-line message + next step + a **Resume** button;
  - **budget stop:** the numbers + **"Resume with a budget of $0.0200"**.
- **The video:** a normal `<video>` player that **seeks**, through Range requests (206).

### The two live runs (watched in a real browser)
1. **"night food stalls in the old town of Bukhara"** (research, hero clip "ask"):
   - the researcher, writer and director ran;
   - **paused at the question** → "Yes" → **queued behind run 2**;
   - **edge-tts lost its internet connection** (`getaddrinfo failed`), shown as the error card;
   - **resumed from the page the next day**: only the missing voices were made;
   - Critic **6 → revise scene 1 → 6** → max rounds;
   - 8 calls + 1 embedding, **$0.0245 list price, $0 billed**.
2. **"why Tashkent metro stations look like palaces"** ($0.01 budget, no revisions):
   - thumbnails appearing one by one;
   - an **ffmpeg failure in the editor** (E4);
   - resumed from the page, then **stopped over budget before the Critic**;
   - **"Resume with $0.0200"** → Critic 5.8 → done;
   - $0.0132.

The Critic's free-tier calls took **359, 373 and 140 seconds**. Without the pulsing timer, the page would have
looked frozen.

---

## 15. Reconnect, replay, budget stop and resume from the browser

### A real SSE stream, with curl (run 2, reconnecting after event 35)
```
> curl -N http://127.0.0.1:8010/runs/2026-10-25_144704/events -H "Last-Event-ID: 35"
: run 2026-10-25_144704
retry: 3000

event: run_started
data: {"id":36,"at":"2026-10-26T12:36:47.488445Z","type":"run_started","resumed":true,"next":["critic"], ...}
id: 36

event: node_started
data: {"id":37,"type":"node_started","node":"critic","round":0,"doing":"watching 4 frames and judging the video","estimate_usd":0.007401, ...}
id: 37

: keep-alive
```
- **Only the events after 35 came back.** That's `Last-Event-ID` doing its job.
- `retry: 3000` tells the browser to wait 3 s before reconnecting.
- `: keep-alive` is the heartbeat, sent after 15 quiet seconds while the Critic waited in Google's queue.
- **After `done`, a reconnect gets 204 No Content.** That's how SSE says "it's finished, stop reconnecting".
  Without it, the browser would reconnect forever.

### Replay
- **A reload during the Critic** rebuilt the whole page in under a second, from event 1, and **the Critic's timer
  kept counting from the event's own start time**.
- **A finished run opened fresh** replays to "done · replayed".
- Both go through the same reducer as a live run (D10), so the three can't look different.

### What the API refuses (real answers)
```
POST /runs {"topic":"tea","research":true}  (Docker stopped) -> 503 {"detail":{"message":"can't connect to the pgvector database ...","hint":"start Docker Desktop, then `docker compose up -d db` ..."}}
POST /runs {"topic":"tea","media":"real"}                    -> 422 "Extra inputs are not permitted"   ← real media can't even be asked for
POST /runs/…/approval  (run already finished)                 -> 409 "not waiting for an answer (finished)"
GET  /runs/…/files/..%2F..%2F.env                             -> 404                                      ← path traversal refused
GET  /runs/…/files/final.mp4   Range: bytes=0-1023           -> 206, content-range: bytes 0-1023/4358943
```

---

## 16. Errors that happened, and what they teach
| # | What happened | Lesson |
|---|---|---|
| E1 | `TypeError: 'EventSourceResponse' object is not iterable` | FastAPI 0.161 now has **built-in SSE** with its own rules (a route with that response class must *yield*). Tutorials using the older `sse-starlette` don't apply. Check the installed version |
| **E2** | **the web could have made PAID images**: `MOCK_MEDIA=1` was set once at startup, but the image code reads it at the moment it makes an image, so a later change would win | found **only by a test that changed it mid-flight**; now every job sets it itself. **A process-wide setting must be set where it's used, every time** |
| **E3** | a running run sometimes showed "crashed" (a flaky test, 3 of 4) | a **check-then-act race**: reading the checkpoint before the worker's state. Read the live source first |
| E4 | `ffmpeg … libx264 … Generic error in an external library` mid-run | not reproduced (probably memory: a 7.8 GB machine with Docker, Next, the API and a browser open). The error message cut off ffmpeg's first lines, which hold the cause; now it keeps them |
| E5 | `next dev` wrote an `AGENTS.md` file into `web/` | Next 16 does that; `agentRules: false` turns it off |
| **E6** | **every failed run's page said "reconnecting…"** although the stream was fine | **the browser treats an SSE event named `error` as its own error event.** `open`, `error` and `message` are reserved names, so the type is `failed`. Found **only in a real browser**: curl and the tests were happy |
| E7 | `--port` became `--PORT` | Typer's metavar quirk again (Phase 7, E3) |
| E8 | TypeScript said a defaulted field was required in the request | openapi-typescript treats defaults as "always present": right for responses, strict for requests |
| E10 | edge-tts DNS failure mid-run | real networks fail; the page showed it clearly, and resume made only what was missing |
| **E11** | **the database volume was gone**: `docker compose up` said `Volume … Created` | something (a `down -v`, or a Docker data reset) deleted it between sessions. **Restored for free** from `chroma/` with the migrate command (0 embedding calls). That's exactly why Chroma is still kept |
| E12 | `ConnectionResetError: [WinError 10054]` in the log | the video player cancels range reads it no longer needs; harmless Windows noise |
| **E14** | **two pages were invisible to git**: `.gitignore`'s `runs/` also matched `web/app/runs/` | found by reading `git status --untracked-files=all` before committing; fixed with `/runs/` (repo root only). **Always look at what a commit will contain** |

---

## 17. Tests, the exercise, and what Phase 9 changes

- **Tests:**
  - **474 Python tests**: 452 pass offline, plus 22 pgvector tests that need the database.
  - **test_api.py: 24 new.** They use FastAPI's `TestClient` (which starts the real worker thread) and Phase 6's
    fake LLM and media, with a real LangGraph in a temp folder. Checked:
    - a whole run: event ids with no gaps, the right order, `event:` matching `type`, and the wire format byte for
      byte;
    - approval pause → POST → continues;
    - budget stop → resume;
    - `Last-Event-ID` replay and 204;
    - traversal refused; Range 206;
    - one run at a time;
    - restart recovery;
    - no secrets in any answer;
    - **media always mock from the web** (E2);
    - **no event named `open`/`error`/`message`** (E6).
  - **Frontend:** `npm run build`, `npm run typecheck`, and `npm test` (9 reducer tests, using Node's built-in
    test runner, so no extra test framework).
- **Optional exercise:** `exercises/phase-8/sse_wire.py`, 20 tests:
  1. write an SSE event;
  2. parse a stream back into events;
  3. replay the events after a `Last-Event-ID`.
  ```bash
  uv run pytest exercises/phase-8
  ```
- **What Phase 9 must change for a public demo:**
  - **Who may start a run:** a login or API keys, rate limits per user and IP, and a **daily global spend cap** (the
    per-run budget exists; a daily total doesn't).
  - **CORS** for the real frontend domain, and **HTTPS** everywhere.
  - **Storage:** runs' media in blob storage; events, jobs, the ledger and checkpoints in **Postgres** (LangGraph
    has a Postgres checkpointer; the knowledge base is already there). Streams wake on Postgres LISTEN/NOTIFY or
    Redis instead of a file.
  - **A real job queue** (Arq or RQ on Redis, or Celery), with workers as separate processes.
  - **Hosting:** Vercel for `web/`, Azure for the API + worker.
- **Phase 8's cost:** 11 LLM calls + 1 embedding, **$0.0377 list price, $0 billed** (3 of the day's 20 Flash
  requests).

---

## Check questions (optional, answer whenever you like)
1. Why does `POST /runs` return 202 instead of waiting for the video?
2. When would you choose SSE, WebSockets or polling? Why SSE here?
3. What does an SSE event look like on the wire, and what's the heartbeat for?
4. How does `Last-Event-ID` let a reloaded page continue without missing or repeating events?
5. What is event sourcing, and how is the run page's reducer like LangGraph's reducers?
6. Why does the browser block the API without CORS headers? Does CORS protect the server?
7. What is a Range request, and what breaks without it?
8. How do you stop `/files/../../.env`?
9. What's the difference between props and state in React? Why don't you update the screen by hand?
10. Why generate TypeScript types from OpenAPI instead of writing them twice?
11. Why is the API local-only and mock-only in this phase?

**Our build**
12. Why does a paused run (waiting for your Yes/No) not block the queue?
13. Why does the events endpoint answer 204 after `done`?
14. Why did every failed run's page say "reconnecting…" (E6), and why didn't curl or the tests notice?
15. How could the web have made paid images (E2), and what's the general rule?
16. Why read the worker's state before the checkpoint (E3)?
17. What two checks keep the TypeScript types in sync with the Python models?
18. The database volume disappeared (E11). How was it restored without a single embedding call?

## Interview angles
- "Long jobs run in a background worker; the API returns 202 with an id, and the browser follows progress over
  Server-Sent Events, with events persisted so a reconnect resumes from Last-Event-ID and finished runs replay."
- "The UI state is a reducer over the event stream, so a live run and a replay render identically."
- "Front and back share one contract: TypeScript types generated from FastAPI's OpenAPI schema."
- "I chose SSE over WebSockets because updates are one-way and SSE is plain HTTP with built-in reconnect."
- "A real-browser test found what curl and unit tests couldn't: an SSE event named `error` is dispatched as the
  EventSource's own error. I renamed it and added a test that forbids reserved names."
- "A test that changed the environment mid-flight showed the web could have made paid images; the fix sets the
  safety setting per job, where it's used."
