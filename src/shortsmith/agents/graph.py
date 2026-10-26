"""The whole pipeline as a LangGraph graph (Steps 6.2-6.7).

    START --research?--> researcher --> writer --> director --> approve_hero --> producer --> editor --> critic
          `------------------------------^                         ^                                     |
                                                                    `--- revise <-- weak scenes, rounds left
                                                                                    otherwise --> END

- researcher, writer, director, critic, revise: one or two LLM calls each (agents/*.py).
- approve_hero: a human-in-the-loop interrupt before the paid hero clip (6.6).
- producer, editor: plain code (no LLM): media through the tools, then ffmpeg.
- critic -> revise -> approve_hero -> producer -> editor -> critic: the evaluator-optimizer loop
  (6.4). Only scenes whose words or picture changed are made again; at most `max_rounds` revisions.

Every node goes through `node()`: it tags the ledger with the run, can be crashed on purpose (the
resume proof), checks the run's budget (Phase 7: budget.py), opens a tracing span, and adds its wall
time to its usage line. The state is checkpointed in SQLite after every node (durability="sync":
written before the next node starts), with thread_id = run_id, so a resume continues where a run stopped.

Run it from the repo root (Phase 7's `shortsmith make` / `shortsmith resume` drive the same graph):
    uv run --env-file .env python -m shortsmith.make "coffee shop in Tashkent" --graph --research --approve yes
    uv run --env-file .env python -m shortsmith.make --resume 2026-10-23_221500 --approve no
"""

import functools
import json
import os
import shutil
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from shortsmith import budget, ledger, llm, tracing
from shortsmith.agents import critic, director, producer, researcher, reviser, writer
from shortsmith.agents.state import RunState, Usage
from shortsmith.media import ffmpeg, images, tts
from shortsmith.models import CriticReport, HookIdea, ResearchNotes, Scene, SceneIssue, Script, ScriptChecks

CHECKPOINTS = Path("checkpoints/graph.sqlite")  # gitignored; one thread per run
CRASH_ENV = "SHORTSMITH_CRASH_BEFORE"  # demo: kill the process when this node is about to start
hard_exit: Callable[[int], None] = os._exit  # like `kill -9`: no cleanup, no finally (tests replace it)

# Told when a node is about to start (after its budget check passed), with its cost estimate: the
# CLI's spinner. A context variable like ledger.CURRENT_RUN: drive() sets it, no node signature changes.
NodeStart = Callable[[str, budget.Estimate], None]
ON_START: ContextVar[NodeStart | None] = ContextVar("shortsmith_on_start", default=None)


# --- the node wrapper ------------------------------------------------------------------------------


def maybe_crash(name: str) -> None:
    """The injected crash of the resume proof: the process dies as if killed, mid-run."""
    if os.environ.get(CRASH_ENV) == name:
        print(f"\n!! {CRASH_ENV}={name}: the process dies now, before '{name}' starts", file=sys.stderr, flush=True)
        hard_exit(70)


def brief(value: Any) -> Any:
    """A short, readable version of a state value for a trace span (never the whole thing)."""
    if isinstance(value, Script):
        return {"title": value.title, "hook": value.hook, "scenes": len(value.scenes), "words": writer.words(value)}
    if isinstance(value, CriticReport):
        return {"round": value.round, "score": value.score, "frames": value.frames,
                "weak": [f"{i.scene} ({i.problem}): {i.reason}" for i in value.weak_scenes]}
    if isinstance(value, BaseModel):
        return brief(value.model_dump(mode="json"))
    if isinstance(value, list):
        return [brief(v) for v in value[:6]] + ([f"... {len(value) - 6} more"] if len(value) > 6 else [])
    if isinstance(value, dict):
        return {key: brief(item) for key, item in value.items()}
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    return value


def reads(name: str, s: RunState) -> dict[str, Any]:
    """What each node actually reads: context engineering, made visible in the trace."""
    match name:
        case "researcher":
            return {"topic": s.topic, "language": s.language, "brand": s.brand}
        case "writer":
            return {"topic": s.topic, "language": s.language, "prompt_version": s.prompt_version,
                    "notes_tokens": llm.estimate_tokens(s.research.for_writer()) if s.research else 0}
        case "director":
            return {"chosen": brief(s.chosen), "visual_style": s.research.visual_style if s.research else ""}
        case "critic" | "revise":
            return {"round": s.round, "plan": brief(s.plan),
                    "weak": brief(s.critic_rounds[-1]) if name == "revise" and s.critic_rounds else None}
        case _:
            return {"round": s.round, "plan": brief(s.plan), "hero": brief(s.hero)}


def node(name: str) -> Callable[[Callable[[RunState], dict]], Callable[[RunState], dict]]:
    """Wrap a node: ledger run tag, crash injection, the budget check, a tracing span, wall time added to its usage."""

    def wrap(fn: Callable[[RunState], dict]) -> Callable[[RunState], dict]:
        @functools.wraps(fn)
        def run(state: RunState) -> dict:
            ledger.CURRENT_RUN.set(state.run_id)  # every ledger line from here on says which run paid
            maybe_crash(name)
            estimate = budget.check(name, state)  # BudgetExceeded here: nothing sent, the last checkpoint stays
            if (on_start := ON_START.get()) is not None:
                on_start(name, estimate)
            start = time.perf_counter()
            paused: GraphInterrupt | None = None
            update: dict = {}
            with tracing.span(name, "AGENT" if name in LLM_NODES else "CHAIN", inputs=reads(name, state)) as span:
                try:
                    update = fn(state)
                except GraphInterrupt as e:  # a pause for a human, not a failure: close the span normally
                    paused = e
                    span.set_attributes({"interrupted": True})
                if paused is None:
                    usage = Usage.model_validate(update.get("usage", {}).get(name, Usage()))
                    usage = usage.model_copy(update={"seconds": round(time.perf_counter() - start, 2)})
                    update["usage"] = {**update.get("usage", {}), name: usage}
                    span.set_outputs(brief({k: v for k, v in update.items() if k != "usage"}))
                    span.set_attributes(usage.model_dump())
            if paused is not None:
                raise paused
            return update

        return run

    return wrap


LLM_NODES = {"researcher", "writer", "director", "critic", "revise"}


# --- the nodes ---------------------------------------------------------------------------------------


@node("researcher")
def researcher_node(s: RunState) -> dict:
    found = llm.paced(lambda: researcher.research(s.topic, s.language, s.brand))
    (s.folder / "research.json").write_text(found.model_dump_json(indent=2), encoding="utf-8")
    return {"research": found.notes, "usage": {"researcher": Usage(
        calls=1, prompt_tokens=found.prompt_tokens, completion_tokens=found.completion_tokens, cost_usd=found.cost_usd)}}


@node("writer")
def writer_node(s: RunState) -> dict:
    result = writer.write(s.topic, s.language, s.research, s.prompt_version)
    (s.folder / "options.json").write_text(json.dumps({
        "prompt_version": result.version, "rejected": result.rejected,
        "pick": result.pick.model_dump() if result.pick else None,
        "options": [o.model_dump() for o in result.options]}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"script_options": result.options, "rejected_options": result.rejected, "pick": result.pick,
            "chosen": result.chosen, "usage": {"writer": Usage.of(result.results)}}


@node("director")
def director_node(s: RunState) -> dict:
    plan, hero, raw, result = director.direct(s.chosen, s.research.visual_style if s.research else "")
    (s.folder / "director.json").write_text(raw.model_dump_json(indent=2), encoding="utf-8")  # what the model wrote
    if s.hero_clips == 0:  # --hero-clips 0: Ken Burns everywhere, and the approval step never asks
        plan, hero = director.ken_burns_only(plan, hero), None
    return {"plan": plan, "hero": hero, "usage": {"director": Usage.of([result])}}


class HeroQuestion(BaseModel):
    """What the approval interrupt shows (and what `make` prints before asking)."""

    question: str
    scene: int
    motion: str
    cost_usd: float
    mock: bool


def is_yes(answer: Any) -> bool:
    return answer is True or str(answer).strip().lower() in {"y", "yes", "true", "1", "ok", "ha"}


@node("approve_hero")
def approve_hero_node(s: RunState) -> dict:
    """Ask before every paid clip: on the first pass, and again only if a revision changed the hero image.
    interrupt() pauses the graph here (state saved); Command(resume=answer) re-runs this node, and
    interrupt() then returns the answer instead of pausing. So nothing before it may have side effects."""
    if not producer.clip_needed(s.folder, s.plan, s.hero):
        return {}
    hero = s.hero
    scene = next(sc for sc in s.plan.scenes if sc.idx == hero.scene)
    mock = images.mock_media()
    price = f"~${hero.cost_usd:.2f}" + (" with real media; MOCK_MEDIA=1 makes a free placeholder clip now" if mock else "")
    question = HeroQuestion(
        question=f"Scene {hero.scene} (\"{scene.narration[:70]}\") as a 5-second AI clip ({hero.motion_prompt}) "
                 f"costs {price}. Continue?",
        scene=hero.scene, motion=hero.motion_prompt, cost_usd=hero.cost_usd, mock=mock)
    answer = interrupt(question.model_dump())
    approved = is_yes(answer)
    update: dict = {"hero": hero.model_copy(update={"approved": approved})}
    if not approved:  # still a finished video: that scene gets the director's Ken Burns move
        scenes = [sc.model_copy(update={"motion": hero.fallback}) if sc.idx == hero.scene else sc for sc in s.plan.scenes]
        update["plan"] = s.plan.model_copy(update={"scenes": scenes})
    return update


@node("producer")
def producer_node(s: RunState) -> dict:
    with producer.tool_caller(s.folder, s.via_mcp) as call_tool:
        made = producer.produce(s.folder, s.plan, s.hero, call_tool, s.round)
    return {"made": [f"round {s.round}: {made}"]}


@node("editor")
def editor_node(s: RunState) -> dict:
    voiceovers = tts.load_voiceovers(s.folder)
    if voiceovers is None or len(voiceovers) != len(s.plan.scenes):
        raise FileNotFoundError(f"{s.folder}: not every scene has its voice yet")
    final = s.folder / "final.mp4"
    if final.exists() and s.round > 0:  # keep the video this round replaces, for before/after
        (s.folder / "history" / f"round_{s.round}").mkdir(parents=True, exist_ok=True)
        shutil.copy(final, s.folder / "history" / f"round_{s.round}" / "final.mp4")
    ffmpeg.assemble(s.folder, s.plan, voiceovers)
    return {"final_path": str(final)}


@node("critic")
def critic_node(s: RunState) -> dict:
    voiceovers = tts.load_voiceovers(s.folder)
    last_asked = s.critic_rounds[-1].round if s.critic_rounds else None
    focus = [r.scene for r in s.revisions if r.round == last_asked and r.changed]  # look at what just changed
    frames = critic.run_frames(s.folder, s.plan, voiceovers, focus, s.round)
    report, result = critic.review(s.topic, s.plan, s.research, frames, s.round, mock=images.mock_media())
    (s.folder / "critic" / f"round_{s.round}" / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"critic_rounds": [report], "usage": {"critic": Usage.of([result])}}


@node("revise")
def revise_node(s: RunState) -> dict:
    last = s.critic_rounds[-1]
    plan, revisions, result = reviser.revise(s.plan, last.weak_scenes, last.round)
    return {"plan": plan, "revisions": revisions, "usage": {"revise": Usage.of([result])}}


# --- the edges -----------------------------------------------------------------------------------------


def route_start(s: RunState) -> str:
    return "researcher" if s.research_on else "writer"


def route_after_critic(s: RunState) -> str:
    """The evaluator-optimizer loop's exit: done when the critic is happy or nothing is flagged,
    and at most max_rounds revisions (critic_rounds holds one report per review: max_rounds + 1)."""
    last = s.critic_rounds[-1]
    if last.score >= s.pass_score or not last.weak_scenes:
        return END
    if len(s.critic_rounds) > s.max_rounds:
        return END
    return "revise"


def stop_reason(s: RunState) -> str:
    if not s.critic_rounds:
        return "not reviewed"
    last = s.critic_rounds[-1]
    if last.score >= s.pass_score:
        return f"passed ({last.score:g} >= {s.pass_score:g})"
    if not last.weak_scenes:
        return f"nothing flagged (score {last.score:g})"
    return f"max rounds ({s.max_rounds} revisions; last score {last.score:g})"


NODES: dict[str, Callable[[RunState], dict]] = {
    "researcher": researcher_node, "writer": writer_node, "director": director_node,
    "approve_hero": approve_hero_node, "producer": producer_node, "editor": editor_node,
    "critic": critic_node, "revise": revise_node,
}


def build(nodes: dict[str, Callable[[RunState], dict]] | None = None) -> StateGraph:
    """The graph's wiring. `nodes` replaces some nodes (tests swap in fakes; the edges stay)."""
    nodes = NODES | (nodes or {})
    graph = StateGraph(RunState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_conditional_edges(START, route_start, ["researcher", "writer"])
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "director")
    graph.add_edge("director", "approve_hero")
    graph.add_edge("approve_hero", "producer")
    graph.add_edge("producer", "editor")
    graph.add_edge("editor", "critic")
    graph.add_conditional_edges("critic", route_after_critic, ["revise", END])
    graph.add_edge("revise", "approve_hero")
    return graph


# Checkpoints are deserialised back into these classes. By default LangGraph imports whatever
# class a checkpoint names (a checkpoint file someone else can write = code they can run), so
# the classes are listed explicitly: anything else comes back as plain data, with a warning.
SAFE_TYPES = [RunState, Usage, Script, Scene, ResearchNotes, HookIdea, CriticReport, SceneIssue, ScriptChecks,
              writer.WriterPick, writer.OptionScore, director.HeroPlan, reviser.Revision]


@contextmanager
def open_graph(db: Path = CHECKPOINTS, nodes: dict[str, Callable[[RunState], dict]] | None = None,
               ) -> Iterator[CompiledStateGraph]:
    """The compiled graph with its SQLite checkpointer. One connection per process; closed on exit."""
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, check_same_thread=False)  # the saver has its own lock
    try:
        saver = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=SAFE_TYPES))
        yield build(nodes).compile(checkpointer=saver)
    finally:
        conn.close()


def config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


Approver = Callable[[dict], Any]  # the interrupt's value -> the answer, or None to stop here (paused)


def drive(app: CompiledStateGraph, payload: Any, run_id: str, approve: Approver | None = None,
          on_update: Callable[[str, dict], None] | None = None, on_start: NodeStart | None = None,
          on_made: producer.MadeHook | None = None) -> dict:
    """Run (or resume) until END or an unanswered interrupt; returns the state's values.

    stream(stream_mode="updates") yields {node: what it returned} after each node: that is the
    progress the CLI prints. An interrupt ends the stream; the answer goes back in as Command(resume=...).
    A node's exception (BudgetExceeded included) passes through; its checkpoint is the last finished node.
    on_start / on_made reach inside the nodes (a node starts; the producer made a file) through context
    variables: the web API's live events (Phase 8) use them, the CLI uses on_start for its spinner.
    """
    token = ON_START.set(on_start)
    made_token = producer.ON_MADE.set(on_made)
    try:
        while True:
            for update in app.stream(payload, config(run_id), stream_mode="updates", durability="sync"):
                for name, values in update.items():
                    if name != "__interrupt__" and on_update:
                        on_update(name, values or {})
            snapshot = app.get_state(config(run_id))
            if not snapshot.next:
                return snapshot.values  # reached END
            if not snapshot.interrupts:
                raise RuntimeError(f"the graph stopped before {snapshot.next} without an interrupt")
            answer = approve(snapshot.interrupts[0].value) if approve else None
            if answer is None:
                return snapshot.values  # paused: resume later with an answer
            payload = Command(resume=answer)
    finally:
        producer.ON_MADE.reset(made_token)
        ON_START.reset(token)


def new_run_id(runs: Path = Path("runs")) -> Path:
    """A fresh run folder named after the time; a second run in the same second gets a suffix."""
    base = runs / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = base.with_name(f"{base.name}-{n}")
    return run_dir


def new_state(topic: str, language: str = "en", brand: str | None = None, research: bool = False,
              prompt_version: str = writer.DEFAULT_VERSION, max_rounds: int = 2, pass_score: float = 7.0,
              via_mcp: bool = False, runs: Path = Path("runs"), budget_usd: float | None = None,
              hero_clips: int = 1, media: Literal["mock", "real"] = "mock") -> RunState:
    run_dir = new_run_id(runs)
    run_dir.mkdir(parents=True)
    return RunState(run_id=run_dir.name, run_dir=str(run_dir), topic=topic, language=language, brand=brand,
                    research_on=research or brand is not None, prompt_version=prompt_version,
                    max_rounds=max_rounds, pass_score=pass_score, via_mcp=via_mcp, budget_usd=budget_usd,
                    hero_clips=hero_clips, media=media)


# --- the command line (make --graph / make --resume) -------------------------------------------------


def describe(name: str, update: dict) -> str:
    """What a finished node decided, in one line (the legacy console line and the Phase 7 CLI share it)."""
    match name:
        case "researcher":
            return f"notes: {len(update['research'].hooks)} hook patterns, {len(update['research'].angles)} angles"
        case "writer":
            pick = update.get("pick")
            return (f"{len(update['script_options'])} options, picked {pick.best if pick else 1}: {update['chosen'].hook!r}"
                    + (f" | rejected: {len(update['rejected_options'])}" if update.get("rejected_options") else ""))
        case "director":
            hero = update.get("hero")
            return f"hero = scene {hero.scene}: {hero.motion_prompt}" if hero else "no hero clip (--hero-clips 0)"
        case "approve_hero":
            hero = update.get("hero")
            return (f"hero clip {'approved' if hero.approved else 'refused: Ken Burns instead'}" if hero
                    else "no new clip needed")
        case "producer":
            return update["made"][0]
        case "editor":
            return update["final_path"]
        case "critic":
            report = update["critic_rounds"][0]
            return f"round {report.round}: score {report.score:g}, weak {report.weak or 'none'} (frames of scenes {report.frames})"
        case "revise":
            return "; ".join(f"scene {r.scene} {r.problem}{'' if r.changed else ' (UNCHANGED)'}" for r in update["revisions"])
        case _:
            return ""


def _progress(name: str, update: dict) -> None:
    """One console line per finished node: time, tokens, and what it decided."""
    usage = Usage.model_validate(update.get("usage", {}).get(name, Usage()))
    tokens = f"{usage.prompt_tokens:>6} in {usage.completion_tokens:>5} out" if usage.calls else f"{'':>19}"
    print(f"{name:<13}{usage.seconds:>7.1f} s | {tokens} | {describe(name, update)}", flush=True)


def approver(answer: str | None) -> Approver:
    """--approve answers every approval question; without it, ask when there is a terminal, else stop."""

    def ask(question: dict) -> str | None:
        print(f"\n?? {question['question']}")
        if answer is not None:
            print(f"   answered by --approve: {answer}\n")
            return answer
        if sys.stdin.isatty():  # on Windows the NUL device counts as a tty too: hence the EOFError below
            try:
                reply = input("   continue? [y/N] ").strip() or "no"
                print()
                return reply
            except EOFError:
                print()
        print("   no one to ask: the run is saved and paused here.")
        return None

    return ask


def run_cli(args: Any) -> None:
    """make --graph / --resume: run the graph, print each node as it finishes, then a summary."""
    from shortsmith import make  # the summary table; imported here because make imports this module lazily

    if not args.no_trace:
        tracing.setup()
    if args.crash_before:
        os.environ[CRASH_ENV] = args.crash_before
    with open_graph() as app:
        if args.resume:
            run_id = args.resume
            snapshot = app.get_state(config(run_id))
            if not snapshot.values:
                raise SystemExit(f"no checkpoints for run {run_id!r} in {CHECKPOINTS}")
            if not snapshot.next:
                raise SystemExit(f"run {run_id} already finished: {snapshot.values.get('final_path')}")
            payload, topic = None, snapshot.values["topic"]
            print(f"resuming {run_id} before {', '.join(snapshot.next)} ({topic!r})")
        else:
            state = new_state(args.topic, args.lang, args.brand, args.research, args.prompt, args.max_rounds,
                              args.pass_score, args.mcp)
            run_id, payload, topic = state.run_id, state.model_dump(), state.topic
            print(f"run {run_id} | {topic!r} ({state.language}) | prompt {state.prompt_version} | "
                  f"max {state.max_rounds} revisions, pass at {state.pass_score:g} | "
                  f"images: {'mock placeholders' if images.mock_media() else images.source().label + ', PAID'}"
                  f"{' | tools over MCP' if state.via_mcp else ''}")
        before = len(ledger.entries(ledger.DEFAULT_PATH, run_id))
        start = time.perf_counter()
        with tracing.span(f"run {run_id}", "AGENT", inputs={"topic": topic, "resume": bool(args.resume)},
                          tags={"run_id": run_id, "resumed": str(bool(args.resume)).lower()}) as root:
            values = drive(app, payload, run_id, approver(args.approve), _progress)
            root.set_outputs({"final_path": values.get("final_path"), "next": list(app.get_state(config(run_id)).next)})
        s = RunState.model_validate(values)
        paused = bool(app.get_state(config(run_id)).next)
    lines = ledger.entries(ledger.DEFAULT_PATH, run_id)
    print(f"\n{'paused' if paused else 'done'} in {time.perf_counter() - start:.0f} s | ledger lines for this run: "
          f"{len(lines)} ({len(lines) - before} new) | {sum(e.prompt_tokens + e.completion_tokens for e in lines):,} tokens, "
          f"${sum(e.cost_usd for e in lines):.6f} list price"
          + (f" | trace {root.trace_id}" if root.trace_id else ""))
    if paused:
        print(f"resume with:  uv run --env-file .env python -m shortsmith.make --resume {run_id} --approve yes|no")
        return
    print(f"stop: {stop_reason(s)} | scores by round: {[r.score for r in s.critic_rounds]}")
    voiceovers = tts.load_voiceovers(s.folder)
    print(f"\n{s.plan.title!r} ({s.plan.language})\n")
    print(make.summary(s.plan, voiceovers, Path(s.final_path)))
