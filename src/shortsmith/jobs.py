"""Runs in the background (Step 8.1): a queue, one worker thread, and the events a run sends.

A request never runs the graph. POST /runs saves a job and answers at once; the worker thread takes
the jobs one at a time, drives the same graph as the CLI (graph.drive, the same node wrapper and
budget check), and turns the graph's hooks into events (events.py) that the page streams:

    on_start (graph.ON_START)     -> node_started             the node wrapper, after its budget check
    on_made  (producer.ON_MADE)   -> media_made               after each image, voice and clip
    on_update (drive's stream)    -> node_finished, video_made, critic_report, revision
    approve (the interrupt)       -> approval_needed (+ approval_answered when the run's policy answers)
    what drive() ends with        -> done | budget_stopped | error   (or a pause at the question)

Why one thread and a FIFO, and not the alternatives:
- FastAPI's BackgroundTasks run after the response, in the server's thread pool: no queue (two
  clicks = two runs at once), no place in line to show, no "is it still running?", and a job is
  simply gone after a restart. Right for "send an email after the response", not for a 10-minute run.
- One worker thread with our own queue = a pool with max_workers=1, plus the order and positions the
  page shows. One, because the free tier allows a few requests a minute and 20 Flash requests a day:
  two runs at once would only make both slower and hit 429s.
- A real job queue (Celery, RQ or Arq: a broker such as Redis plus separate worker processes) once jobs
  must survive the web process dying mid-run, run on other machines, retry by themselves, or once more
  than one web process takes requests (Phase 9, if the public demo gets traffic).

Durability without a broker: the job is a file (runs/<id>/job.json), the state is LangGraph's
checkpoint, the events are a file. After a restart, queued jobs are queued again in their order, and a
run this server was driving is marked "interrupted": its checkpoint has every finished node, so it resumes.
"""

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from shortsmith import budget, costs, ledger, llm, runs, tracing
from shortsmith.agents import critic, graph
from shortsmith.agents.state import RunState, Usage
from shortsmith.agents.tools import audio_file, image_file
from shortsmith.console import DOING
from shortsmith.events import (ApprovalAnswered, ApprovalNeeded, ApprovePolicy, BudgetStopped, CriticReported, Done,
                               EventBase, EventLog, Failed, MediaMade, NodeFinished, NodeStarted, Queued, Revised,
                               RunStarted, SceneChange, VideoMade, now, settles)
from shortsmith.media.ffmpeg import ai_clip_file

JOB_FILE = "job.json"
MAX_QUEUED = 5  # waiting runs at most: a local tool, and each run spends part of the day's free quota

log = logging.getLogger("shortsmith.jobs")


class Job(BaseModel):
    """One piece of work for the worker, saved as runs/<id>/job.json (so a restart finds it again)."""

    run_id: str
    kind: Literal["start", "resume"]
    approve: ApprovePolicy = "ask"  # the run's answer to the hero-clip question; kept for its resumes
    answer: Literal["yes", "no"] | None = None  # resume a paused run with this answer
    budget_usd: float | None = None  # resume with this budget (written into the checkpoint, like the CLI's)
    state: dict[str, Any] | None = None  # start: the new run's RunState
    queued_at: datetime = Field(default_factory=now)
    status: Literal["queued", "running", "settled"] = "queued"


def save_job(job: Job, runs_dir: Path) -> None:
    (runs_dir / job.run_id / JOB_FILE).write_text(job.model_dump_json(indent=2), encoding="utf-8")


def load_job(folder: Path) -> Job | None:
    path = folder / JOB_FILE
    try:
        return Job.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
    except ValueError as e:
        log.warning("%s: can't read it: %s", path, e)
        return None


class QueueFull(RuntimeError):
    pass


class QueueView(BaseModel):
    running: str | None  # the run the worker is on
    node: str | None  # what it's doing
    queued: list[str]  # waiting, first = next
    max_queued: int


class Worker:
    """One run at a time: a FIFO of jobs and the one thread that works through it."""

    def __init__(self, execute: Callable[[Job], None], max_queued: int = MAX_QUEUED):
        self._execute = execute
        self._pending: deque[Job] = deque()
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self.current: Job | None = None
        self.node: str | None = None  # the current run's node (for the status of the run)
        self.max_queued = max_queued

    def start(self) -> None:
        self._stopping = False
        # A daemon thread: stopping the server doesn't wait for a 10-minute run. The run's checkpoint
        # keeps every finished node, and the next start marks it interrupted (Jobs.recover).
        self._thread = threading.Thread(target=self._loop, name="shortsmith-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def submit(self, job: Job, limit: bool = True) -> list[str]:
        """Queue a job; returns the run ids ahead of it (the running one first). QueueFull when full."""
        with self._cv:
            if limit and len(self._pending) >= self.max_queued:
                raise QueueFull(f"{len(self._pending)} runs are already waiting (at most {self.max_queued})")
            ahead = ([self.current.run_id] if self.current else []) + [j.run_id for j in self._pending]
            self._pending.append(job)
            self._cv.notify_all()
            return ahead

    def ahead_of(self, run_id: str) -> list[str] | None:
        """The runs before this one in the queue; None if it isn't waiting."""
        with self._cv:
            ids = [j.run_id for j in self._pending]
            if run_id not in ids:
                return None
            return ([self.current.run_id] if self.current else []) + ids[: ids.index(run_id)]

    def is_running(self, run_id: str) -> bool:
        current = self.current
        return current is not None and current.run_id == run_id

    def active(self, run_id: str) -> bool:
        return self.is_running(run_id) or self.ahead_of(run_id) is not None

    def view(self) -> QueueView:
        with self._cv:
            return QueueView(running=self.current.run_id if self.current else None, node=self.node,
                             queued=[j.run_id for j in self._pending], max_queued=self.max_queued)

    def wait_idle(self, timeout: float) -> bool:
        """Block until nothing is running or waiting (tests; a clean shutdown)."""
        with self._cv:
            return self._cv.wait_for(lambda: self.current is None and not self._pending, timeout)

    def _loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._pending or self._stopping)
                if self._stopping:
                    return
                self.current, self.node = self._pending.popleft(), None
            try:
                self._execute(self.current)
            except BaseException:  # a job must never take the worker down with it
                log.exception("run %s: the worker caught an error", self.current.run_id)
            finally:
                with self._cv:
                    self.current, self.node = None, None
                    self._cv.notify_all()


class Emitter:
    """One job's graph hooks -> events in the run's log."""

    def __init__(self, events: EventLog, state: RunState, policy: ApprovePolicy, on_node: Callable[[str], None]):
        self.events, self.run_id, self.state = events, state.run_id, state
        self.reports = list(state.critic_rounds)  # so the critic's event can say what the edge decides
        self.round = len(self.reports)  # = RunState.round: 0 until the first review
        self.policy, self.on_node = policy, on_node
        self.node: str | None = None
        self.video_url: str | None = None  # the last cut's URL (its round, not the one after the critic)

    def emit(self, event: EventBase) -> None:
        self.events.append(self.run_id, event)

    def url(self, path: str) -> str:
        return f"/runs/{self.run_id}/files/{path}?v={self.round}"

    def started(self, name: str, estimate: budget.Estimate) -> None:
        self.node = name
        self.on_node(name)
        self.emit(NodeStarted(node=name, round=self.round, doing=DOING.get(name, "running"),
                              estimate_usd=estimate.usd, basis=estimate.basis))

    def finished(self, name: str, update: dict) -> None:
        usage = Usage.model_validate(update.get("usage", {}).get(name, Usage()))
        self.emit(NodeFinished(node=name, round=self.round, seconds=usage.seconds, calls=usage.calls,
                               prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                               cost_usd=usage.cost_usd, summary=graph.describe(name, update)))
        if name == "editor":
            self.video_url = self.url("final.mp4")
            self.emit(VideoMade(round=self.round, url=self.video_url))
        elif name == "critic":
            self._critic(update["critic_rounds"][0])
        elif name == "revise":
            self.emit(Revised(round=self.round - 1, scenes=[
                SceneChange(**r.model_dump(exclude={"round"}), changed=r.changed) for r in update["revisions"]]))

    def _critic(self, report: Any) -> None:
        self.reports.append(report)
        s = self.state.model_copy(update={"critic_rounds": self.reports})
        then = "done" if graph.route_after_critic(s) == graph.END else "revise"
        reason = graph.stop_reason(s) if then == "done" else f"revising scene(s) {', '.join(map(str, report.weak))}"
        self.emit(CriticReported(
            round=report.round, score=report.score, pass_score=s.pass_score, summary=report.summary, hook=report.hook,
            weak_scenes=report.weak_scenes, frames=report.frames, model=report.model, then=then, reason=reason,
            frame_urls=[self.url(f"critic/round_{report.round}/scene_{i:02d}.jpg") for i in report.frames]))
        self.round += 1

    def made(self, kind: str, scene: int) -> None:
        path = {"image": image_file, "voice": audio_file, "clip": ai_clip_file}[kind](scene).as_posix()
        self.emit(MediaMade(kind=kind, scene=scene, round=self.round, path=path, url=self.url(path)))

    def approve(self, question: dict) -> str | None:
        """graph.drive's approver: "ask" pauses the run (None); "yes"/"no" answer at once."""
        waiting = self.policy == "ask"
        self.emit(ApprovalNeeded(**question, waiting=waiting))
        if waiting:
            return None
        self.emit(ApprovalAnswered(answer=self.policy, by="policy"))
        return self.policy


class Jobs:
    """The API's one object for background work: queue a job, say where a run stands, recover after a restart."""

    def __init__(self, runs_dir: Path = runs.RUNS, trace: bool = True, max_queued: int = MAX_QUEUED):
        self.runs = runs_dir
        self.events = EventLog(runs_dir)
        self.trace = trace
        self._traced = False
        self.worker = Worker(self._run, max_queued)

    # --- for the API ---

    def submit(self, job: Job, limit: bool = True) -> Queued:
        save_job(job, self.runs)
        ahead = self.worker.submit(job, limit)
        return self.events.append(job.run_id, Queued(position=len(ahead) + 1, ahead=ahead))

    def live(self, run_id: str, view: QueueView) -> runs.RunStatus | None:
        """The worker's word on a run (running / queued), or None: then the checkpoint's status holds.

        runs.status_of() can't see this process's own worker (a run's `.running` marker holds the
        server's process id, which never counts as "another process"), so the API asks here. Take the
        `view` BEFORE reading the checkpoint: read the other way round, a run that finishes in between
        looks "crashed" (an old checkpoint, and a worker that is no longer on it)."""
        if view.running == run_id:
            return runs.RunStatus(kind="running", node=view.node or "start", detail="the web server is working on it")
        if run_id in view.queued:
            ahead = ([view.running] if view.running else []) + view.queued[: view.queued.index(run_id)]
            return runs.RunStatus(kind="queued", detail=f"{len(ahead)} run(s) ahead" if ahead else "starts next")
        return None

    def recover(self) -> list[str]:
        """After a (re)start: runs this server was driving when it stopped are marked interrupted, and
        queued jobs are queued again, oldest first. Only web runs (with a job.json) are touched; a CLI
        process's run is left to the CLI."""
        waiting: list[Job] = []
        for folder in sorted(p for p in self.runs.iterdir() if p.is_dir()) if self.runs.is_dir() else []:
            job = load_job(folder)
            if job is None:
                continue
            marker = folder / runs.RUNNING
            if marker.exists() and runs.running_pid(folder) is not None:
                continue  # another live process (a CLI resume) is driving it
            if job.status == "running":
                last = self.events.last(folder.name)
                node = last.node if isinstance(last, NodeStarted) else None
                if not settles(last):
                    self.events.append(folder.name, Failed(
                        node=node, interrupted=True,
                        message=f"the server stopped while this run was {f'at {node}' if node else 'starting'}",
                        hint="resume it: everything up to its last finished node is saved"))
                job.status = "settled"
                save_job(job, self.runs)
            marker.unlink(missing_ok=True)
            if job.status == "queued":
                waiting.append(job)
        for job in sorted(waiting, key=lambda j: j.queued_at):
            self.submit(job, limit=False)
        return [j.run_id for j in waiting]

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    # --- in the worker thread ---

    def _run(self, job: Job) -> None:
        job.status = "running"
        save_job(job, self.runs)
        # Mock media for every web run, set again per job: images.mock_media() reads the environment when an
        # image is made, and nothing the web sends can change it (`--real` stays a CLI-only, confirmed choice).
        os.environ["MOCK_MEDIA"] = "1"
        try:
            if self.trace and not self._traced:
                tracing.setup()  # MLflow's import takes seconds: on the first job, not at server start
                self._traced = True
            with graph.open_graph() as app, runs.marked_running(self.runs / job.run_id):
                self._drive(app, job)
        except Exception as e:  # before the graph ran (no checkpoint to read, a broken job file)
            log.exception("run %s failed before the graph started", job.run_id)
            self.events.append(job.run_id, Failed(node=None, message=f"{type(e).__name__}: {e}", interrupted=False,
                                                  hint="see the server's log"))
        finally:
            job.status = "settled"
            save_job(job, self.runs)

    def _drive(self, app: Any, job: Job) -> None:
        from shortsmith import cli  # explain(): the CLI's one-line message + next step, word for word

        config = graph.config(job.run_id)
        snapshot = app.get_state(config)
        if not snapshot.values:  # a new run (or one whose first node never finished: start it again)
            s = RunState.model_validate(job.state)
            payload: Any = s.model_dump()
        else:
            s = RunState.model_validate(snapshot.values)
            if job.budget_usd is not None and job.budget_usd != s.budget_usd:  # as `shortsmith resume --budget`
                app.update_state(config, {"budget_usd": job.budget_usd})
                s = s.model_copy(update={"budget_usd": job.budget_usd})
            payload = graph.Command(resume=job.answer) if job.answer else None
        resumed = bool(snapshot.values)
        history = ledger.entries(ledger.DEFAULT_PATH)
        provider_name = llm.provider_name()
        self.events.append(job.run_id, RunStarted(
            resumed=resumed, next=list(snapshot.next), topic=s.topic, language=s.language, research=s.research_on,
            brand=s.brand, max_rounds=s.max_rounds, pass_score=s.pass_score, hero_clips=s.hero_clips,
            approve=job.approve, budget_usd=s.budget_usd, spent_usd=budget.spent(s.run_id, history), media=s.media,
            provider=provider_name, model=llm.model_for("writer"), critic_model=critic.critic_model(),
            free_tier=provider_name == "gemini" and costs.free_tier(), billing=costs.billing_note()))
        emitter = Emitter(self.events, s, job.approve, self._at_node)
        log.info("run %s: %s", job.run_id, f"resumed before {', '.join(snapshot.next)}" if resumed else "started")
        start = time.perf_counter()
        stopped: budget.BudgetExceeded | None = None
        failure: Exception | None = None
        with tracing.span(f"run {job.run_id}", "AGENT", inputs={"resume": resumed, "source": "web"},
                          tags={"run_id": job.run_id, "resumed": str(resumed).lower(), "source": "web"}) as root:
            try:
                graph.drive(app, payload, job.run_id, emitter.approve, on_update=emitter.finished,
                            on_start=emitter.started, on_made=emitter.made)
            except budget.BudgetExceeded as e:  # a clean stop: the last finished node's checkpoint stays
                stopped = e
            except Exception as e:  # a failed node: the same, plus an error to explain
                failure = e
            snapshot = app.get_state(config)
            root.set_outputs({"next": list(snapshot.next), "stopped": str(stopped or failure or "") or None})
        s = RunState.model_validate(snapshot.values) if snapshot.values else s
        history = ledger.entries(ledger.DEFAULT_PATH)
        if stopped:
            self.events.append(job.run_id, BudgetStopped(
                node=stopped.node, spent_usd=stopped.spent, estimate_usd=stopped.estimate, limit_usd=stopped.limit,
                suggested_usd=budget.suggested(s, history), message=str(stopped)))
            log.info("run %s: %s", job.run_id, stopped)
        elif failure:
            message, hint, _ = cli.explain(failure, job.run_id)
            self.events.append(job.run_id, Failed(node=emitter.node, message=message, hint=hint, interrupted=False))
            log.warning("run %s failed at %s: %s%s", job.run_id, emitter.node, message,
                        "".join(f"\n{note}" for note in getattr(failure, "__notes__", [])))  # e.g. ffmpeg's whole stderr
        elif snapshot.next:
            log.info("run %s: paused at the hero-clip question", job.run_id)  # the page has the question already
        else:
            lines = [e for e in history if e.run_id == job.run_id]
            final = self.runs / job.run_id / "final.mp4"
            self.events.append(job.run_id, Done(
                final_url=(emitter.video_url or emitter.url("final.mp4")) if final.exists() else None,
                scores=[r.score for r in s.critic_rounds], stop_reason=graph.stop_reason(s),
                seconds=round(time.perf_counter() - start, 1), calls=len(lines),
                tokens=sum(e.prompt_tokens + e.completion_tokens + e.reasoning_tokens for e in lines),
                cost_usd=sum(e.cost_usd for e in lines), billed_usd=costs.Billing(history).total(lines),
                budget_usd=s.budget_usd))
            log.info("run %s: done (%s)", job.run_id, graph.stop_reason(s))

    def _at_node(self, name: str) -> None:
        self.worker.node = name
