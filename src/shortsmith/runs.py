"""The runs on disk, as data (Step 7.1): what `shortsmith list` and `show` print, and what Phase 8's
web API can serve. Read-only: nothing here changes a run.

A run is a folder in runs/ whose name is the run id. Three sources are combined:
- the folder: title and language (script.json), the critic's reports (critic/round_N/report.json),
  the video's length (voiceovers.json: the editor's own frame counts, so no ffprobe per run), the
  files, and reruns.json;
- the checkpoint (checkpoints/graph.sqlite, thread_id = run id): where the run stands (finished,
  paused at the approval question, over budget, crashed at a node), its settings, the revisions and
  the time per node. The checkpoint is the truth about status; the one thing it can't tell, running
  or crashed, comes from a `.running` marker with the id of the process driving the run;
- the ledger (runs/ledger.jsonl): what the run cost, from the lines tagged with its id. Runs from
  before Phase 6 have no tagged lines: their cost is unknown, not $0.
MLflow trace ids come from mlflow.db's own tables by plain read-only SQL (importing mlflow takes seconds).
"""

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, computed_field

from shortsmith import costs, ledger
from shortsmith.agents.director import HeroPlan
from shortsmith.agents.reviser import Revision
from shortsmith.agents.state import RunState
from shortsmith.media import ffmpeg, tts
from shortsmith.models import CriticReport, Script
from shortsmith.rerun import RerunRecord, load_records

RUNS = Path("runs")
CHECKPOINTS = Path("checkpoints/graph.sqlite")  # = graph.CHECKPOINTS (not imported: LangGraph only when needed)
MLFLOW_DB = Path("mlflow.db")
RUN_FILES = ("script.json", "options.json", "research.json", "media.json")  # any of these = a run folder

# Ledger steps -> the graph node that makes them ("agent": media tools record their own lines under it).
STEP_NODE = {"researcher": "researcher", "rag": "researcher", "writer": "writer", "writer-pick": "writer",
             "director": "director", "critic": "critic", "revise": "revise", "agent": "producer", "producer": "producer"}
NODE_ORDER = ["researcher", "writer", "director", "approve_hero", "producer", "editor", "critic", "revise"]

log = logging.getLogger("shortsmith.runs")


class RunNotFound(LookupError):
    """No run matches what the person typed."""


class RunStatus(BaseModel):
    kind: Literal["finished", "running", "paused", "over_budget", "crashed", "no_checkpoint"]
    node: str | None = None  # where the run stands: the node it continues with
    detail: str = ""  # the approval question, the budget numbers, or the error

    @computed_field
    @property
    def label(self) -> str:
        match self.kind:
            case "finished":
                return "finished"
            case "running":
                return f"running ({self.node})"
            case "paused":
                return "paused at approval"
            case "over_budget":
                return f"over budget before {self.node}"
            case "crashed":
                return f"crashed at {self.node}"
            case _:
                return "no checkpoint"


class RunSummary(BaseModel):
    """One row of `shortsmith list`."""

    id: str
    folder: str
    created: datetime | None  # from the folder name
    topic: str | None
    title: str | None
    language: str | None
    score: float | None  # the critic's last score
    reviews: int  # critic rounds so far
    length_s: float | None  # final.mp4's length (None: no video yet)
    final_mp4: str | None  # absolute path
    calls: int  # ledger lines tagged with this run
    tokens: int
    cost_usd: float | None  # list price; None = no tagged ledger lines (runs before Phase 6)
    billed_usd: float | None
    status: RunStatus


class Settings(BaseModel):
    prompt_version: str
    research_on: bool
    brand: str | None
    max_rounds: int
    pass_score: float
    via_mcp: bool
    budget_usd: float | None
    hero_clips: int
    media: str


class SceneRow(BaseModel):
    idx: int
    motion: str
    words: int
    voice_s: float | None  # the voice file's real length
    narration: str
    visual_prompt: str


class NodeCost(BaseModel):
    node: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0  # visible output + thinking
    cost_usd: float = 0.0  # list price, from the ledger
    billed_usd: float = 0.0
    seconds: float | None = None  # wall time, from the checkpoint's usage (None: not a graph node)


class TraceInfo(BaseModel):
    trace_id: str
    started: datetime
    seconds: float | None
    status: str


class Files(BaseModel):
    final_mp4: str | None
    video: ffmpeg.VideoInfo | None  # what ffprobe says about final.mp4
    images: int
    voices: int
    hero_clips: list[str]
    critic_frames: int
    history: list[str]  # history/round_N, history/rerun_N: what each round or rerun replaced
    saved: list[str]  # the JSON files the run left (script.json, options.json, ...)


class RunDetail(RunSummary):
    """Everything `shortsmith show` prints."""

    settings: Settings | None  # None: a run without a checkpoint
    scenes: list[SceneRow]
    hook: str | None
    cta: str | None
    hero: HeroPlan | None
    critic_rounds: list[CriticReport]
    revisions: list[Revision]
    made: list[str]  # each producer pass, e.g. "round 1: images 3"
    reruns: list[RerunRecord]
    nodes: list[NodeCost]
    files: Files
    traces: list[TraceInfo]


# --- finding runs -------------------------------------------------------------------------------------


def checkpoint_threads(db: Path = CHECKPOINTS) -> set[str]:
    """The run ids that have a checkpoint, by plain SQL (no LangGraph import)."""
    if not db.exists():
        return set()
    try:
        with closing(sqlite3.connect(db)) as conn:
            return {row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    except sqlite3.Error as e:
        log.warning("could not read %s: %s", db, e)
        return set()


def run_folders(runs: Path = RUNS, threads: set[str] | None = None) -> list[Path]:
    """Every run folder, newest first (folder names are timestamps)."""
    if not runs.is_dir():
        return []
    threads = checkpoint_threads() if threads is None else threads
    found = [p for p in runs.iterdir() if p.is_dir() and (p.name in threads or any((p / f).exists() for f in RUN_FILES))]
    return sorted(found, key=lambda p: p.name, reverse=True)


def resolve(ref: str, runs: Path = RUNS) -> Path:
    """A run from what a person types: its id, "latest", a unique start of an id, or a folder path."""
    folders = run_folders(runs)
    if ref in ("latest", "last"):
        if not folders:
            raise RunNotFound(f"there are no runs in {runs}/ yet: make one with `shortsmith make \"topic\"`")
        return folders[0]
    for folder in folders:
        if folder.name == ref:
            return folder
    path = Path(ref)
    if path.is_dir() and any((path / f).exists() for f in RUN_FILES):
        return path
    matches = [f for f in folders if f.name.startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise RunNotFound(f"{ref!r} matches {len(matches)} runs ({', '.join(m.name for m in matches[:5])}): type more of the id")
    raise RunNotFound(f"no run {ref!r} in {runs}/: see `shortsmith list`")


def snapshots(run_ids: list[str], db: Path = CHECKPOINTS) -> dict[str, Any]:
    """The latest checkpoint of each run that has one. LangGraph (~3 s to import) only when there is one."""
    threads = checkpoint_threads(db)
    wanted = [r for r in run_ids if r in threads]
    if not wanted:
        return {}
    from shortsmith.agents import graph

    with graph.open_graph(db) as app:
        return {r: app.get_state(graph.config(r)) for r in wanted}


# --- running or crashed? -------------------------------------------------------------------------------
# A checkpoint looks the same while a process is still working on the next node and after that
# process died: next = that node, no error. So the CLI leaves a marker in the run folder while it
# drives a run, with its process id; a marker whose process is gone was left by a crash.

RUNNING = ".running"


def pid_alive(pid: int) -> bool:
    """Is this process still running? (Windows has no os.kill(pid, 0): ask the kernel for its exit code.)"""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, it just isn't ours
        return True
    return True


@contextmanager
def marked_running(folder: Path) -> Iterator[None]:
    """`.running` in the run folder while this process drives the run (removed even on an error;
    a hard kill leaves it, with a process id that is gone)."""
    path = folder / RUNNING
    path.write_text(json.dumps({"pid": os.getpid(), "since": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def running_pid(folder: Path) -> int | None:
    """The process driving this run right now, if another one is (this process never counts)."""
    try:
        pid = int(json.loads((folder / RUNNING).read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return pid if pid != os.getpid() and pid_alive(pid) else None


def status_of(snapshot: Any | None, running: int | None = None) -> RunStatus:
    """Where a run stands, read from its checkpoint (a LangGraph StateSnapshot); `running`: the id of
    a live process that is driving it (running_pid)."""
    if running is not None and (snapshot is None or not snapshot.values or snapshot.next):
        node = snapshot.next[0] if snapshot is not None and snapshot.next else "start"
        return RunStatus(kind="running", node=node, detail=f"process {running} is working on it")
    if snapshot is None or not snapshot.values:
        return RunStatus(kind="no_checkpoint")
    if not snapshot.next:
        return RunStatus(kind="finished")
    node = snapshot.next[0]
    if snapshot.interrupts:
        return RunStatus(kind="paused", node=node, detail=str(snapshot.interrupts[0].value.get("question", "")))
    errors = [str(t.error).strip('"') for t in snapshot.tasks if t.error]  # LangGraph keeps a node's exception as its repr
    if errors and errors[0].startswith("BudgetExceeded("):
        return RunStatus(kind="over_budget", node=node, detail=errors[0].split("(", 1)[1].strip("')"))
    if errors:
        return RunStatus(kind="crashed", node=node, detail=errors[0][:300])
    return RunStatus(kind="crashed", node=node,
                     detail="the process stopped without an error (killed, or still running in another window)")


# --- reading one run ------------------------------------------------------------------------------------


def _created(name: str) -> datetime | None:
    try:
        return datetime.strptime(name[:17], "%Y-%m-%d_%H%M%S")
    except ValueError:
        return None


def _script(folder: Path, state: RunState | None) -> Script | None:
    """The script as it is on disk (a rerun changes it), else the graph's plan or pick."""
    path = folder / "script.json"
    if path.exists():
        return Script.load(path)
    return (state.plan or state.chosen) if state else None


def _reports(folder: Path) -> list[CriticReport]:
    paths = sorted(folder.glob("critic/round_*/report.json"), key=lambda p: int(p.parent.name.split("_")[1]))
    return [CriticReport.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


def _length(folder: Path) -> float | None:
    if not (folder / "final.mp4").exists():
        return None
    voiceovers = tts.load_voiceovers(folder)
    return round(sum(ffmpeg.scene_frames(voiceovers)) / ffmpeg.FPS, 1) if voiceovers else None


def _topic(folder: Path, state: RunState | None) -> str | None:
    if state:
        return state.topic
    research = folder / "research.json"
    if research.exists():
        return json.loads(research.read_text(encoding="utf-8")).get("topic")
    return None


def summary(folder: Path, snapshot: Any | None, lines: list[ledger.LedgerEntry]) -> RunSummary:
    """One run's row. `lines`: this run's ledger lines."""
    state = RunState.model_validate(snapshot.values) if snapshot is not None and snapshot.values else None
    script = _script(folder, state)
    rounds = state.critic_rounds if state and state.critic_rounds else _reports(folder)
    final = folder / "final.mp4"
    return RunSummary(
        id=folder.name, folder=str(folder), created=_created(folder.name), topic=_topic(folder, state),
        title=script.title if script else None, language=script.language if script else (state.language if state else None),
        score=rounds[-1].score if rounds else None, reviews=len(rounds), length_s=_length(folder),
        final_mp4=str(final.resolve()) if final.exists() else None, calls=len(lines),
        tokens=sum(e.prompt_tokens + e.completion_tokens + e.reasoning_tokens for e in lines),
        cost_usd=sum(e.cost_usd for e in lines) if lines else None,
        billed_usd=sum(costs.billed_usd(e) for e in lines) if lines else None,
        status=status_of(snapshot, running_pid(folder)),
    )


def list_runs(limit: int | None = None, runs: Path = RUNS, db: Path = CHECKPOINTS,
              ledger_path: Path = ledger.DEFAULT_PATH) -> list[RunSummary]:
    """Every run, newest first (at most `limit`)."""
    folders = run_folders(runs, checkpoint_threads(db))
    if limit:
        folders = folders[:limit]
    states = snapshots([f.name for f in folders], db)
    by_run: dict[str, list[ledger.LedgerEntry]] = {}
    for entry in ledger.entries(ledger_path):
        if entry.run_id:
            by_run.setdefault(entry.run_id, []).append(entry)
    return [summary(f, states.get(f.name), by_run.get(f.name, [])) for f in folders]


def traces(run_id: str, db: Path = MLFLOW_DB) -> list[TraceInfo]:
    """The MLflow traces tagged with this run (a resume is a new trace with the same run_id tag)."""
    if not db.exists():
        return []
    try:
        with closing(sqlite3.connect(db)) as conn:
            rows = conn.execute(
                "SELECT i.request_id, i.timestamp_ms, i.execution_time_ms, i.status FROM trace_info i "
                "JOIN trace_tags t ON t.request_id = i.request_id WHERE t.key = 'run_id' AND t.value = ? "
                "ORDER BY i.timestamp_ms", (run_id,)).fetchall()
    except sqlite3.Error as e:  # an MLflow version with other tables: the ids are missing, nothing else
        log.warning("could not read the traces in %s: %s", db, e)
        return []
    return [TraceInfo(trace_id=r[0], started=datetime.fromtimestamp(r[1] / 1000, timezone.utc),
                      seconds=round(r[2] / 1000, 1) if r[2] is not None else None, status=str(r[3])) for r in rows]


def _files(folder: Path) -> Files:
    final = folder / "final.mp4"
    video = None
    if final.exists():
        try:
            video = ffmpeg.probe(final)
        except (OSError, ValueError) as e:  # no ffprobe on PATH, or a broken file: the rest still shows
            log.warning("ffprobe could not read %s: %s", final, e)
    return Files(
        final_mp4=str(final.resolve()) if final.exists() else None, video=video,
        images=len(list(folder.glob("scenes/scene_*.png"))), voices=len(list(folder.glob("audio/scene_*.mp3"))),
        hero_clips=sorted(p.relative_to(folder).as_posix() for p in folder.glob(f"{ffmpeg.AI_CLIPS}/*.mp4")),
        critic_frames=len(list(folder.glob("critic/round_*/scene_*.jpg"))),
        history=sorted(p.relative_to(folder).as_posix() for p in folder.glob("history/*") if p.is_dir()),
        saved=sorted(p.name for p in folder.glob("*.json")),
    )


def node_costs(lines: list[ledger.LedgerEntry], state: RunState | None) -> list[NodeCost]:
    """Cost per node from the ledger (the money), time per node from the checkpoint's usage."""
    rows: dict[str, NodeCost] = {}
    for e in lines:
        name = STEP_NODE.get(e.step, e.step)
        row = rows.setdefault(name, NodeCost(node=name))
        row.calls += 1
        row.prompt_tokens += e.prompt_tokens
        row.completion_tokens += e.completion_tokens + e.reasoning_tokens
        row.cost_usd += e.cost_usd
        row.billed_usd += costs.billed_usd(e)
    for name, usage in (state.usage if state else {}).items():
        rows.setdefault(name, NodeCost(node=name)).seconds = usage.seconds
    order = {name: n for n, name in enumerate(NODE_ORDER)}
    return sorted(rows.values(), key=lambda r: order.get(r.node, len(order)))


def detail(folder: Path, snapshot: Any | None = None, ledger_path: Path = ledger.DEFAULT_PATH,
           mlflow_db: Path = MLFLOW_DB) -> RunDetail:
    """Everything known about one run."""
    lines = ledger.entries(ledger_path, folder.name)
    head = summary(folder, snapshot, lines)
    state = RunState.model_validate(snapshot.values) if snapshot is not None and snapshot.values else None
    script = _script(folder, state)
    voiceovers = tts.load_voiceovers(folder) or []
    voice_s = {s.idx: v.duration_s for s, v in zip(script.scenes, voiceovers)} if script and len(voiceovers) == len(script.scenes) else {}
    settings = Settings(**state.model_dump(include=set(Settings.model_fields))) if state else None
    return RunDetail(
        **head.model_dump(exclude={"status"}), status=head.status, settings=settings,
        scenes=[SceneRow(idx=s.idx, motion=s.motion, words=len(s.narration.split()), voice_s=voice_s.get(s.idx),
                         narration=s.narration, visual_prompt=s.visual_prompt) for s in (script.scenes if script else [])],
        hook=script.hook if script else None, cta=script.cta if script else None,
        hero=state.hero if state else None,
        critic_rounds=state.critic_rounds if state and state.critic_rounds else _reports(folder),
        revisions=state.revisions if state else [], made=state.made if state else [],
        reruns=load_records(folder), nodes=node_costs(lines, state), files=_files(folder), traces=traces(folder.name, mlflow_db),
    )
