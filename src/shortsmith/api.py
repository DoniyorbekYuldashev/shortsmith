"""The web API (Step 8.1): FastAPI over the same graph, runs, ledger and budget the CLI uses.

    shortsmith serve                    http://127.0.0.1:8010   (OpenAPI docs at /docs)
    cd web && npm run dev               http://127.0.0.1:3000   (the Next.js page, web/)

    POST /runs                          start a run: queued, 202 + its id
    GET  /runs                          the runs, newest first (runs.RunSummary, as `shortsmith list --json`)
    GET  /runs/{id}                     one run (runs.RunDetail, as `shortsmith show --json`)
    GET  /runs/{id}/events              Server-Sent Events: the run's progress, live and replayed (Last-Event-ID)
    GET  /runs/{id}/log                 the same events as one JSON list
    POST /runs/{id}/approval            answer the hero-clip question of a paused run
    POST /runs/{id}/resume              continue a stopped run: after a budget stop (with a bigger budget) or a crash
    GET  /runs/{id}/files/{path}        a run's images, voices, clips, critic frames and final.mp4 (Range requests)
    GET  /costs                         costs.CostReport (as `shortsmith cost --json`)
    GET  /queue                         what the worker is doing and who waits
    GET  /options                       languages and brands, for the form
    GET  /health                        provider, keys set or not, media, ffmpeg, the vector store, the worker

The work runs in jobs.py's worker thread, one run at a time; a request only checks, queues and reads.
Boundaries in this phase: it listens on 127.0.0.1 only and has no login; media is ALWAYS mock (no paid
image or clip can be started from here, whatever MOCK_MEDIA says); CORS allows only the local frontend;
files come only from inside a run folder; no response or log line holds a key or the database password.
A public demo (Phase 9) needs auth or rate limits, a spend cap, HTTPS and a real job queue first.
"""

import asyncio
import copy
import os
import re
import shutil
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, format_sse_event
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shortsmith import budget, costs, ledger, llm, providers, runs
from shortsmith.events import ApprovalAnswered, ApprovePolicy, Event, Queued
from shortsmith.jobs import Job, Jobs, QueueFull, QueueView, load_job
from shortsmith.research import stores

HOST = "127.0.0.1"  # local only in Phase 8: nothing on the network can reach it
PORT = 8010  # 8000 is the owner's other project; 5432/5433 are Postgres; 3000 is the Next.js dev server
WEB_ORIGINS = ("http://127.0.0.1:3000", "http://localhost:3000")  # CORS: the local frontend only
BRANDS = Path("data/hooks/brands")
RUN_ID = re.compile(r"[0-9A-Za-z][0-9A-Za-z_-]*")  # a run folder's name: no dots or slashes, so never a path
RETRY_MS = 3000  # how long a browser waits before reconnecting a dropped stream
POLL_S = 0.3  # how often the stream looks for new events (a dict lookup; the file is read only when there are)
HEARTBEAT_S = 15.0  # an SSE comment line when nothing happened for this long, so idle streams aren't cut

# What a run folder may serve, and as what. Everything else (job.json, events.jsonl, the marker) is not a file
# for the browser: its data comes from the JSON endpoints.
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".mp4": "video/mp4",
               ".mp3": "audio/mpeg", ".srt": "text/plain; charset=utf-8"}


# --- request and response models ------------------------------------------------------------------------


class Problem(BaseModel):
    """Every refusal: one line on what is wrong, one on what to do (as the CLI's error + next)."""

    message: str
    hint: str = ""


class ProblemResponse(BaseModel):
    detail: Problem


def problem(status: int, message: str, hint: str = "") -> HTTPException:
    return HTTPException(status, detail=Problem(message=message, hint=hint).model_dump())


ERRORS = {code: {"model": ProblemResponse} for code in (400, 404, 409, 429, 503)}


class StartRun(BaseModel):
    """POST /runs. The same choices as `shortsmith make`, minus --real (mock only in this phase)."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=200, description="What the video is about, in a few words.")
    language: Literal["en", "uz", "ru"] = "en"
    research: bool = Field(False, description="YouTube trends + the hook library (needs the database).")
    brand: str | None = Field(None, pattern=r"^[a-z0-9-]+$", description="A brand in data/hooks/brands/; implies research.")
    approve: ApprovePolicy = Field("ask", description="The hero-clip question: ask (pause for a person), yes, or no.")
    max_rounds: int = Field(2, ge=0, le=5, description="Critic -> revise rounds at most.")
    pass_score: float = Field(7.0, ge=0, le=10)
    budget_usd: float | None = Field(None, ge=0, description="Hard limit for the whole run, LIST price.")
    hero_clips: Literal[0, 1] = 1

    @field_validator("topic")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("say in a few words what the video is about")
        return value.strip()


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Literal["yes", "no"]


class ResumeRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_usd: float | None = Field(None, ge=0, description="A new budget (list price), saved with the run.")


class Accepted(BaseModel):
    run_id: str
    status: runs.RunStatus
    queued: Queued  # its place in line (the same event the stream sends first)
    events: str  # the SSE stream's path


class Options(BaseModel):
    languages: list[str]
    brands: list[str]
    defaults: StartRun


class Health(BaseModel):
    ok: bool  # a run without research can start
    version: str
    provider: str
    model: str  # the writer's (and researcher's, director's, reviser's unless set apart)
    critic_model: str
    key_set: bool  # every key the text, critic and embedding models need is set (no value is ever sent)
    youtube_key_set: bool
    media: Literal["mock"]  # always, in this phase
    free_tier: bool  # Gemini free tier: billed $0, the numbers shown are list price
    billing: str  # what "billed" means with these settings, in one line
    setup: providers.Setup | None  # every role's provider and model, keys set or not, allowance settings
    allowance: costs.AllowanceToday | None  # OpenAI's free daily tokens used today, by our ledger
    ffmpeg: bool
    research_ready: bool  # the vector store answers (pgvector: Docker + the database)
    vector_store: stores.StoreStatus | None  # the database URL masked; None if it couldn't be asked
    problems: list[Problem]
    queue: QueueView


# --- the app ----------------------------------------------------------------------------------------------


def brands() -> list[str]:
    return sorted(p.stem for p in BRANDS.glob("*.md"))


def web_origins() -> list[str]:
    extra = os.environ.get("SHORTSMITH_WEB_ORIGINS", "")
    return [o.strip() for o in extra.split(",") if o.strip()] or list(WEB_ORIGINS)


def create_app(trace: bool = True, origins: list[str] | None = None, max_queued: int | None = None) -> FastAPI:
    """The app. Nothing starts until the server does (lifespan): creating it is free (the OpenAPI dump)."""
    jobs = Jobs(trace=trace, **({"max_queued": max_queued} if max_queued is not None else {}))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        os.environ["MOCK_MEDIA"] = "1"  # mock only from the web, whatever .env says (jobs set it again per run)
        jobs.recover()
        jobs.start()
        yield
        jobs.stop()

    app = FastAPI(title="Shortsmith API", version=package_version("shortsmith"), lifespan=lifespan,
                  description="One line of text in, a ~30-second vertical video out. Local only (127.0.0.1), "
                              "mock media only, one run at a time.")
    app.state.jobs = jobs
    app.add_middleware(CORSMiddleware, allow_origins=origins or web_origins(), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "Last-Event-ID"])

    def folder_of(run_id: str) -> Path:
        folder = runs.RUNS / run_id
        if not RUN_ID.fullmatch(run_id) or not folder.is_dir():
            raise problem(404, f"no run {run_id!r}", "GET /runs lists them")
        return folder

    def with_job[R: runs.RunSummary](row: R, view: QueueView) -> R:
        """The worker's word on top of the checkpoint's (running, queued), and a queued run's topic.
        `view` must be taken before the row was read (Jobs.live says why)."""
        update: dict = {"status": jobs.live(row.id, view) or row.status}
        if row.topic is None and (job := load_job(runs.RUNS / row.id)) is not None and job.state:
            update.update(topic=job.state.get("topic"), language=row.language or job.state.get("language"))
        return row.model_copy(update=update)

    def status_now(run_id: str, folder: Path) -> tuple[runs.RunStatus, object | None]:
        view = jobs.worker.view()  # first the worker, then the checkpoint (Jobs.live)
        snapshot = runs.snapshots([run_id]).get(run_id)
        return jobs.live(run_id, view) or runs.status_of(snapshot, runs.running_pid(folder)), snapshot

    def preflight(research: bool) -> None:
        from shortsmith import cli

        try:
            cli.preflight(real=False, research=research)
        except cli.Problem as e:
            raise problem(503, str(e), e.hint) from None
        except stores.StoreUnavailable as e:  # no quiet fallback: say what's down and how to start it
            raise problem(503, e.message, e.hint) from None

    def accepted(job: Job) -> Accepted:
        try:
            queued = jobs.submit(job)
        except QueueFull as e:
            raise problem(429, str(e), "wait for a run to finish (GET /queue)") from None
        status = jobs.live(job.run_id, jobs.worker.view()) or runs.RunStatus(kind="queued", detail="starts next")
        return Accepted(run_id=job.run_id, status=status, queued=queued, events=f"/runs/{job.run_id}/events")

    # --- runs ---

    @app.post("/runs", status_code=202, responses=ERRORS, tags=["runs"])
    def start_run(body: StartRun) -> Accepted:
        """Queue a new run. It starts when the worker is free; follow it at `events`."""
        from shortsmith.agents import graph

        research = body.research or body.brand is not None
        if body.brand is not None and body.brand not in brands():
            raise problem(400, f"no brand {body.brand!r}", f"the brands are: {', '.join(brands()) or 'none'}")
        if len(jobs.worker.view().queued) >= jobs.worker.max_queued:
            raise problem(429, f"{jobs.worker.max_queued} runs are already waiting", "wait for one to finish (GET /queue)")
        preflight(research)  # before a folder exists: a missing key or a stopped database costs nothing
        state = graph.new_state(body.topic, body.language, body.brand, research, max_rounds=body.max_rounds,
                                pass_score=body.pass_score, budget_usd=body.budget_usd, hero_clips=body.hero_clips,
                                media="mock")
        return accepted(Job(run_id=state.run_id, kind="start", approve=body.approve, state=state.model_dump(mode="json")))

    @app.get("/runs", tags=["runs"])
    def list_runs(limit: Annotated[int, Query(ge=0, le=500)] = 50) -> list[runs.RunSummary]:
        """The runs, newest first (0 = all), with the worker's status on top of the checkpoint's."""
        view = jobs.worker.view()
        return [with_job(r, view) for r in runs.list_runs(limit or None)]

    @app.get("/runs/{run_id}", responses=ERRORS, tags=["runs"])
    def show_run(run_id: str) -> runs.RunDetail:
        folder = folder_of(run_id)
        view = jobs.worker.view()
        return with_job(runs.detail(folder, runs.snapshots([run_id]).get(run_id)), view)

    @app.post("/runs/{run_id}/approval", status_code=202, responses=ERRORS, tags=["runs"])
    def answer_approval(run_id: str, body: Answer) -> Accepted:
        """Answer the hero-clip question of a paused run: yes = the clip, no = Ken Burns for that scene."""
        folder = folder_of(run_id)
        status, _ = status_now(run_id, folder)
        if status.kind != "paused":
            raise problem(409, f"run {run_id} is not waiting for an answer ({status.label})")
        if len(jobs.worker.view().queued) >= jobs.worker.max_queued:
            raise problem(429, f"{jobs.worker.max_queued} runs are already waiting", "try again when one has finished")
        old = load_job(folder)
        jobs.events.append(run_id, ApprovalAnswered(answer=body.answer, by="person"))
        return accepted(Job(run_id=run_id, kind="resume", answer=body.answer, approve=old.approve if old else "ask"))

    @app.post("/runs/{run_id}/resume", status_code=202, responses=ERRORS, tags=["runs"])
    def resume_run(run_id: str, body: ResumeRun) -> Accepted:
        """Continue a stopped run from its checkpoint (nothing finished is paid for again): after a budget
        stop, with a bigger budget; after a crash or a server restart, as it was."""
        from shortsmith.agents.state import RunState

        folder = folder_of(run_id)
        if (live := jobs.live(run_id, jobs.worker.view())) is not None:
            raise problem(409, f"run {run_id} is already {live.label}")
        if (pid := runs.running_pid(folder)) is not None:
            raise problem(409, f"run {run_id} is running in another process ({pid})", "wait for it, or stop it there")
        status, snapshot = status_now(run_id, folder)
        old = load_job(folder)
        policy = old.approve if old else "ask"
        if snapshot is None or not snapshot.values:  # queued once, never checkpointed: start it again
            if old is None or old.state is None:
                raise problem(409, f"run {run_id} has no checkpoint, so it can't be resumed",
                              "runs made with --simple or before Phase 6 have none")
            state = old.state | ({"budget_usd": body.budget_usd} if body.budget_usd is not None else {})
            preflight(RunState.model_validate(state).research_on)
            return accepted(Job(run_id=run_id, kind="start", approve=policy, state=state))
        if status.kind == "finished":
            raise problem(409, f"run {run_id} already finished (nothing to do)")
        if status.kind == "paused":
            raise problem(409, f"run {run_id} is waiting for the hero-clip answer", f"POST /runs/{run_id}/approval")
        s = RunState.model_validate(snapshot.values)
        if s.media == "real":
            raise problem(409, f"run {run_id} was made with --real", f"resume it from the CLI: shortsmith resume {run_id}")
        preflight("researcher" in budget.nodes_left(s))
        return accepted(Job(run_id=run_id, kind="resume", budget_usd=body.budget_usd, approve=policy))

    # --- events ---

    def finished(run_id: str) -> bool:
        last = jobs.events.last(run_id)
        return last is not None and last.type == "done" and not jobs.worker.active(run_id)

    async def stream(run_id: str, last: int, follow: bool) -> AsyncIterator[bytes]:
        yield format_sse_event(comment=f"run {run_id}", retry=RETRY_MS)
        quiet, sent = 0.0, ""
        while True:
            for event in jobs.events.after(run_id, last):
                yield format_sse_event(data_str=event.model_dump_json(), event=event.type, id=str(event.id))
                last, quiet, sent = event.id, 0.0, event.type
            if not follow or sent == "done" and not jobs.worker.active(run_id):
                return
            await asyncio.sleep(POLL_S)
            quiet += POLL_S
            if quiet >= HEARTBEAT_S:  # a comment line: EventSource ignores it, proxies see traffic
                yield b": keep-alive\n\n"
                quiet = 0.0

    # Not response_class=EventSourceResponse: FastAPI would then iterate whatever the function returns
    # (it's meant for `yield` endpoints), and this one returns either a stream or a 204.
    @app.get("/runs/{run_id}/events", tags=["events"], responses={
        **ERRORS, 200: {"description": "text/event-stream: `event:` = the type, `data:` = one event of GET "
                                       "/runs/{run_id}/log as JSON, `id:` = its id", "content": {"text/event-stream": {}}},
        204: {"description": "The run finished and every event was sent: stop reconnecting"}})
    def run_events(run_id: str, after: Annotated[int, Query(ge=0)] = 0, follow: bool = True,
                   last_event_id: Annotated[str | None, Header()] = None) -> Response:
        """The run's events as Server-Sent Events: everything after `after` (or the Last-Event-ID header a
        reconnecting browser sends), then new ones as they happen. Ends after `done`; follow=false ends
        after the replay. The event types are the models of GET /runs/{id}/log."""
        folder_of(run_id)
        start = max(after, int(last_event_id) if last_event_id and last_event_id.isdigit() else 0)
        if finished(run_id) and start >= jobs.events.last_id(run_id):
            return Response(status_code=204)  # the SSE way to say "don't reconnect"
        return EventSourceResponse(stream(run_id, start, follow), headers={"X-Accel-Buffering": "no"})

    @app.get("/runs/{run_id}/log", responses=ERRORS, tags=["events"])
    def run_log(run_id: str, after: Annotated[int, Query(ge=0)] = 0) -> list[Event]:
        """The run's events so far, as one JSON list (what the stream would replay)."""
        folder_of(run_id)
        return jobs.events.after(run_id, after)

    # --- files ---

    @app.get("/runs/{run_id}/files/{path:path}", response_class=FileResponse, responses=ERRORS, tags=["files"])
    def run_file(run_id: str, path: str) -> FileResponse:
        """A file inside the run folder (images, voices, clips, critic frames, final.mp4). Range requests
        work (FileResponse), so the browser can seek in the video. Nothing outside the folder, ever."""
        folder = folder_of(run_id).resolve()
        target = (folder / path).resolve()
        if not target.is_relative_to(folder) or target.suffix.lower() not in MEDIA_TYPES or not target.is_file():
            raise problem(404, f"no file {path!r} in run {run_id}")
        # no-cache = "ask before reusing": a cut after a rerun has the same name; the ETag makes asking cheap
        return FileResponse(target, media_type=MEDIA_TYPES[target.suffix.lower()], headers={"Cache-Control": "no-cache"})

    # --- the rest ---

    @app.get("/costs", tags=["costs"])
    def get_costs(days: Annotated[int | None, Query(ge=1)] = None, run: str | None = None) -> costs.CostReport:
        """The ledger by day, run, model and step; list price next to billed; today's free-tier use."""
        from shortsmith.agents import critic

        return costs.report(ledger.entries(ledger.DEFAULT_PATH), days, run, critic_model=critic.critic_model())

    @app.get("/queue", tags=["runs"])
    def queue() -> QueueView:
        return jobs.worker.view()

    @app.get("/options", tags=["setup"])
    def options() -> Options:
        return Options(languages=["en", "uz", "ru"], brands=brands(), defaults=StartRun(topic="your topic"))

    @app.get("/health", tags=["setup"])
    def health() -> Health:
        """What `shortsmith doctor` checks, as data: never a key, the database URL masked."""
        from shortsmith.agents import critic
        from shortsmith.research import rag

        found: list[Problem] = []
        name = os.environ.get("LLM_PROVIDER", "openai")
        setup: providers.Setup | None = None
        model, critic_model, key_set, embedding = "", "", False, ""
        try:
            setup = providers.setup()
            model, critic_model, embedding = llm.model_for("writer"), critic.critic_model(), llm.embedding_model()
            missing = sorted({r.key_env for r in setup.roles if r.role in ("text", "critic", "embeddings")
                              and r.key_env and not r.key_set})
            key_set = not missing
            for key in missing:
                found.append(Problem(message=f"{key} is not set (LLM_PROVIDER={name})",
                                     hint="put it in .env in the repo folder, then restart `shortsmith serve`"))
        except ValueError as e:
            found.append(Problem(message=str(e), hint="LLM_PROVIDER and EMBEDDING_PROVIDER are gemini or openai; "
                                                      "IMAGE_PROVIDER openai, fal or mock"))
        ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
        if not ffmpeg_ok:
            found.append(Problem(message="ffmpeg or ffprobe is not on PATH", hint="install ffmpeg, then restart the server"))
        store: stores.StoreStatus | None = None
        research_ready = False
        try:
            s = store = stores.status(rag.collection_name(embedding, llm.EMBED_DIMENSIONS))
            research_ready = s.backend == "chroma" or bool(s.reachable)
            if not research_ready:
                found.append(Problem(message=f"research is off: {s.error}", hint=s.hint or ""))
        except Exception as e:  # an unknown VECTOR_STORE, a broken install: a status, not a crash
            found.append(Problem(message=f"the vector store can't be checked: {type(e).__name__}: {e}",
                                 hint="`shortsmith doctor` shows more"))
        allowance = None
        try:
            allowance = costs.allowance_today(ledger.entries(ledger.DEFAULT_PATH), datetime.now(timezone.utc))
        except ValueError as e:  # an OPENAI_FREE_* setting we can't read
            found.append(Problem(message=str(e), hint="see .env.example for the OPENAI_FREE_* settings"))
        return Health(ok=key_set and ffmpeg_ok, version=package_version("shortsmith"), provider=name, model=model,
                      critic_model=critic_model, key_set=key_set,
                      youtube_key_set=bool(os.environ.get("YOUTUBE_API_KEY")), media="mock",
                      free_tier=name == "gemini" and costs.free_tier(), billing=setup.billing if setup else "",
                      setup=setup, allowance=allowance, ffmpeg=ffmpeg_ok, research_ready=research_ready,
                      vector_store=store, problems=found, queue=jobs.worker.view())

    return app


# --- running it -----------------------------------------------------------------------------------------------


def serve(port: int = PORT, trace: bool = True) -> None:
    """uvicorn on 127.0.0.1. Its access log goes to stderr too: stdout stays for results (none here)."""
    import uvicorn

    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["handlers"]["access"]["stream"] = "ext://sys.stderr"
    log_config["loggers"]["shortsmith"] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    os.environ["MOCK_MEDIA"] = "1"
    uvicorn.run(create_app(trace=trace), host=HOST, port=port, log_config=log_config)


def openapi_json() -> str:
    import json

    return json.dumps(create_app(trace=False).openapi(), indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":  # uv run python -m shortsmith.api --openapi web/openapi.json  (web/'s `npm run types`)
    if sys.argv[1:2] != ["--openapi"] or len(sys.argv) != 3:
        sys.exit("usage: python -m shortsmith.api --openapi FILE")
    Path(sys.argv[2]).write_text(openapi_json(), encoding="utf-8", newline="\n")
    print(f"wrote {sys.argv[2]}", file=sys.stderr)
