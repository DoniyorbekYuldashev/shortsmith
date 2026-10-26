"""What a run tells the web page while it works (Step 8.1): typed events, saved per run, replayable.

Every event is a Pydantic model with a `type`. The worker (jobs.py) appends them to
runs/<id>/events.jsonl, one JSON line each, with an id that grows by one per run (1, 2, 3, ...).
The SSE stream (api.py) sends each one as

    event: node_finished
    data: {"id": 7, "type": "node_finished", "node": "writer", ...}
    id: 7

so a browser that loses the connection reconnects with `Last-Event-ID: 7` and gets only what came
after, and a page opened after the run finished replays the whole run from the file.

The file is the truth; memory only caches each run's last id. That makes replay free (read the
file), survives a server restart, and lets the SSE endpoint tell "something new" by comparing two
integers instead of re-reading the file every time it looks.
"""

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from shortsmith.models import SceneIssue

EVENTS_FILE = "events.jsonl"
ApprovePolicy = Literal["ask", "yes", "no"]  # how a run answers the hero-clip question: pause and ask, or answer now

log = logging.getLogger("shortsmith.events")


def now() -> datetime:
    return datetime.now(timezone.utc)


class EventBase(BaseModel):
    # In the OpenAPI schema, fields with a default still count as always present in a response, so the
    # generated TypeScript types say `id: number`, not `id?: number`.
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: int = 0  # set by EventLog.append: 1, 2, 3, ... per run
    at: datetime = Field(default_factory=now)


class Queued(EventBase):
    """Accepted, waiting for the worker (one run at a time)."""

    type: Literal["queued"] = "queued"
    position: int  # 1 = starts next
    ahead: list[str]  # the run ids before it (the running one first)


class RunStarted(EventBase):
    """The worker took the run: a new run, or a resume (after an answer, a budget stop or a crash)."""

    type: Literal["run_started"] = "run_started"
    resumed: bool
    next: list[str]  # where a resume continues ([] for a new run)
    topic: str
    language: str
    research: bool
    brand: str | None
    max_rounds: int
    pass_score: float
    hero_clips: int
    approve: ApprovePolicy
    budget_usd: float | None  # list price, for the whole run
    spent_usd: float  # list price the run already spent before this job (a resume)
    media: Literal["mock", "real"]
    provider: str
    model: str
    critic_model: str
    free_tier: bool  # Gemini on the free tier: every call is billed $0 (the numbers are list price)


class NodeStarted(EventBase):
    type: Literal["node_started"] = "node_started"
    node: str
    round: int  # 0 = the first pass; 1 = the first revision round, ...
    doing: str  # what it's doing, for the page while it waits (a free-tier critic call can take minutes)
    estimate_usd: float  # list price expected, from budget.py (the same number the budget is checked with)
    basis: str  # where the estimate comes from


class NodeFinished(EventBase):
    type: Literal["node_finished"] = "node_finished"
    node: str
    round: int
    seconds: float
    calls: int
    prompt_tokens: int
    completion_tokens: int  # visible output + thinking (both billed as output)
    cost_usd: float  # list price of this node's calls
    summary: str  # what it decided, in one line (graph.describe: the CLI's done line)


class MediaMade(EventBase):
    """One file of a producer pass: the page shows the thumbnail now, not after the whole pass."""

    type: Literal["media_made"] = "media_made"
    kind: Literal["image", "voice", "clip"]
    scene: int
    round: int
    path: str  # inside the run folder, e.g. scenes/scene_03.png
    url: str  # /runs/<id>/files/<path>?v=<round>: a remade file gets a new URL, so the browser doesn't show its cached copy


class ApprovalNeeded(EventBase):
    """The hero-clip question (graph.HeroQuestion). With policy "ask" the run pauses here."""

    type: Literal["approval_needed"] = "approval_needed"
    question: str
    scene: int
    motion: str
    cost_usd: float  # with real media; mock media makes a free placeholder clip
    mock: bool
    waiting: bool  # True: paused until a person answers; False: the run's policy answers it


class ApprovalAnswered(EventBase):
    type: Literal["approval_answered"] = "approval_answered"
    answer: Literal["yes", "no"]
    by: Literal["person", "policy"]


class VideoMade(EventBase):
    """The editor cut final.mp4 (again, after each revision round)."""

    type: Literal["video_made"] = "video_made"
    round: int
    url: str


class CriticReported(EventBase):
    type: Literal["critic_report"] = "critic_report"
    round: int
    score: float
    pass_score: float
    summary: str
    hook: str
    weak_scenes: list[SceneIssue]  # worst first, at most 3
    frames: list[int]  # the scenes whose frames the critic saw
    frame_urls: list[str]
    model: str
    then: Literal["revise", "done"]  # what the graph's edge decides next (graph.route_after_critic)
    reason: str  # graph.stop_reason, or "revising scenes 3, 5"


class SceneChange(BaseModel):
    """One revised scene (reviser.Revision, with `changed` spelled out: it's a property there)."""

    scene: int
    problem: str
    reason: str
    fix: str
    changed: bool  # False: the reviser returned the same words and picture
    narration_before: str
    narration_after: str
    visual_before: str
    visual_after: str


class Revised(EventBase):
    type: Literal["revision"] = "revision"
    round: int  # the critic round that asked for it
    scenes: list[SceneChange]


class BudgetStopped(EventBase):
    """Stopped before a node that would take the run over its budget; nothing was sent for it."""

    type: Literal["budget_stopped"] = "budget_stopped"
    node: str
    spent_usd: float
    estimate_usd: float
    limit_usd: float
    suggested_usd: float  # a budget that surely finishes (budget.suggested)
    message: str


class Failed(EventBase):
    """A node failed, or the server stopped mid-run. The checkpoint keeps everything finished: resumable."""

    # Not "error": EventSource dispatches a server message named "error" as its OWN error event (onerror fires
    # and the page thinks the connection dropped). open, error and message are taken (Phase 8 build notes).
    type: Literal["failed"] = "failed"
    node: str | None
    message: str  # the CLI's one-line message (cli.explain)
    hint: str  # what to do next
    interrupted: bool  # True: the server stopped (not an error in the run itself)


class Done(EventBase):
    type: Literal["done"] = "done"
    final_url: str | None
    scores: list[float]  # the critic's score per round
    stop_reason: str  # graph.stop_reason
    seconds: float  # this job's wall time
    calls: int  # the whole run's ledger lines
    tokens: int
    cost_usd: float  # list price
    billed_usd: float  # $0 on the Gemini free tier
    budget_usd: float | None


Event = Annotated[Queued | RunStarted | NodeStarted | NodeFinished | MediaMade | ApprovalNeeded | ApprovalAnswered
                  | VideoMade | CriticReported | Revised | BudgetStopped | Failed | Done, Field(discriminator="type")]
EVENT = TypeAdapter(Event)

# After one of these the run waits for a person (or is over): the worker is free for the next run.
SETTLED = {"done", "approval_needed", "budget_stopped", "failed"}


def settles(event: EventBase | None) -> bool:
    """Is this the last event until someone acts? (An approval answered by the run's policy doesn't wait.)"""
    if event is None:
        return False
    if isinstance(event, ApprovalNeeded):
        return event.waiting
    return getattr(event, "type", "") in SETTLED


class EventLog:
    """runs/<id>/events.jsonl for every run: append (any thread), read what came after an id.

    One lock for the writers: the worker thread appends progress, request threads append
    `queued` and a person's answer, and two writers must never get the same id."""

    def __init__(self, runs: Path):
        self.runs = runs
        self._lock = threading.Lock()
        self._last: dict[str, int] = {}

    def path(self, run_id: str) -> Path:
        return self.runs / run_id / EVENTS_FILE

    def _read(self, run_id: str) -> list[EventBase]:
        path = self.path(run_id)
        if not path.exists():
            return []
        found = []
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.endswith("\n"):  # a line still being written (or cut by a crash): not an event yet
                break
            try:
                found.append(EVENT.validate_json(line))
            except ValueError as e:  # a damaged line must not hide the rest of the run
                log.warning("%s: skipped a line it can't read: %s", path, e)
        return found

    def last_id(self, run_id: str) -> int:
        with self._lock:
            if run_id not in self._last:
                events = self._read(run_id)
                self._last[run_id] = events[-1].id if events else 0
            return self._last[run_id]

    def append(self, run_id: str, event: EventBase) -> EventBase:
        """Give the event the run's next id and add it to the file (written and closed before returning)."""
        self.last_id(run_id)  # loads the cache outside the lock below
        with self._lock:
            event = event.model_copy(update={"id": self._last[run_id] + 1})
            with self.path(run_id).open("a", encoding="utf-8", newline="\n") as file:
                file.write(event.model_dump_json() + "\n")
            self._last[run_id] = event.id
        return event

    def after(self, run_id: str, last_id: int = 0) -> list[EventBase]:
        """The events with an id above `last_id`, in order (all of them for 0)."""
        if last_id >= self.last_id(run_id):
            return []
        return [e for e in self._read(run_id) if e.id > last_id]

    def last(self, run_id: str) -> EventBase | None:
        events = self._read(run_id)
        return events[-1] if events else None
