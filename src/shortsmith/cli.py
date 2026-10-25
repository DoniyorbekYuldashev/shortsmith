"""`shortsmith`: the command line (Phase 7). One line of text in, a ~30-second vertical video out.

    shortsmith make "coffee shop in Tashkent" --research --approve yes
    shortsmith list
    shortsmith show latest
    shortsmith rerun-scene latest 3 --prompt "a wide shot of the bazaar at dawn"
    shortsmith cost --days 7

Typer turns each command function's typed parameters into options, checks and --help; Rich draws the
progress and the tables (console.py). This file only parses, asks and prints: the work happens in
agents/graph.py (runs), rerun.py, runs.py, costs.py and budget.py, which Phase 8's web API can call too.
`python -m shortsmith.make` (the Phase 2-6 command) still works as it was.

Settings, highest first: flags > environment variables > .env in the current folder > defaults.
Exit codes: see ExitCode.
"""

import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from enum import Enum, IntEnum
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.text import Text

from shortsmith import views
from shortsmith.console import NodeProgress, Ui, usd


class ExitCode(IntEnum):
    """The same meaning in every command, so a script can tell outcomes apart."""

    OK = 0
    ERROR = 1  # something failed: the message says what, and what to do next
    USAGE = 2  # bad flags or arguments (Click's own code for usage errors)
    PAUSED = 3  # waiting for an answer to the hero-clip question (state saved: resume with --approve)
    OVER_BUDGET = 4  # stopped before a step that would pass --budget (state saved: resume with a bigger one)
    INTERRUPTED = 130  # Ctrl+C (128 + SIGINT, the shell convention)


EPILOG = (
    "Settings, highest first: flags > environment variables > .env in the current folder > defaults.\n\n"
    "Exit codes: 0 ok, 1 error, 2 usage, 3 paused for an approval, 4 over budget, 130 interrupted.\n\n"
    "Money: every call is priced at list price in runs/ledger.jsonl; on the Gemini free tier it is billed $0. "
    "Images stay free placeholders unless you pass --real."
)

app = typer.Typer(
    name="shortsmith",
    help="Shortsmith: AI agents that turn one line of text into a ~30-second vertical short video.",
    epilog=EPILOG,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,  # our own one-line errors; --debug shows a plain traceback
    context_settings={"help_option_names": ["-h", "--help"]},
)


class Lang(str, Enum):
    en = "en"
    uz = "uz"
    ru = "ru"


class Answer(str, Enum):
    yes = "yes"
    no = "no"


class PromptVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"


@dataclass
class Session:
    """What every command shares: the consoles, --debug, and the run it is working on (for hints)."""

    ui: Ui
    debug: bool = False
    run_id: str | None = None


# Typer 0.27 carries its own copy of Click (typer._click); the `click` package in the venv is another
# one, and its classes never match Typer's. Typer's ClickException is found through BadParameter.
CLICK_ERRORS = tuple(c for c in typer.BadParameter.__mro__ if c.__name__ == "ClickException")


class Problem(Exception):
    """A failure we can explain in one line, with what to do next."""

    def __init__(self, message: str, hint: str = "", code: ExitCode = ExitCode.ERROR):
        super().__init__(message)
        self.hint, self.code = hint, code


# --- errors: one clear message, the next step, a meaningful exit code -------------------------------------


def explain(e: BaseException, run_id: str | None) -> tuple[str, str, ExitCode]:
    """(message, what to do next, exit code) for anything a command can raise."""
    import openai
    from pydantic import ValidationError

    from shortsmith import llm, runs
    from shortsmith.research import stores

    resume = f"shortsmith resume {run_id}" if run_id else ""
    saved = f"the run is saved after its last finished node: {resume}" if run_id else ""
    debug = "run again with --debug to see the traceback"
    match e:
        case Problem():
            return str(e), e.hint, e.code
        case stores.StoreUnavailable():  # never a quiet switch to another store: say what's wrong and what to do
            return e.message, f"{e.hint}; then {resume}" if resume else e.hint, ExitCode.ERROR
        case KeyboardInterrupt():
            return "interrupted", saved, ExitCode.INTERRUPTED
        case runs.RunNotFound():
            return str(e), "`shortsmith list` shows the runs", ExitCode.ERROR
        case llm.DailyQuotaError():
            return (str(e).split("; try another model")[0],
                    f"use another model (--critic-model gemini-3.6-flash for the critic) or wait, then {resume or 'try again'}",
                    ExitCode.ERROR)
        case openai.AuthenticationError() | openai.PermissionDeniedError():
            return ("the LLM provider refused the API key",
                    "check the key in .env (`shortsmith doctor` says which keys are set)", ExitCode.ERROR)
        case openai.RateLimitError():
            return "the LLM provider kept refusing with 429 (rate limit), even after waiting", \
                f"wait a minute, then {resume or 'try again'}", ExitCode.ERROR
        case openai.APIConnectionError():
            return "could not reach the LLM provider", f"check the internet connection, then {resume or 'try again'}", ExitCode.ERROR
        case openai.APIStatusError():
            return f"the LLM provider answered {e.status_code}: {str(e)[:200]}", saved or debug, ExitCode.ERROR
        case ValidationError():
            first = e.errors(include_url=False)[0]["msg"] if e.errors() else str(e)
            return (f"the model's answer broke {e.error_count()} rule(s), e.g.: {first} (the call is in the ledger)",
                    saved or "try again: a new answer usually passes", ExitCode.ERROR)
        case FileNotFoundError() if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            return "ffmpeg or ffprobe is not on PATH", "install ffmpeg, open a new terminal, then `shortsmith doctor`", ExitCode.ERROR
        case subprocess.CalledProcessError():
            notes = " ".join(getattr(e, "__notes__", []))
            return f"{Path(str(e.cmd[0])).name} failed with exit code {e.returncode}. {notes[-300:]}", saved or debug, ExitCode.ERROR
        case ValueError() | FileNotFoundError() | LookupError():
            return str(e), saved or debug, ExitCode.ERROR
        case _:
            return f"{type(e).__name__}: {e}", f"{saved + '; ' if saved else ''}{debug}", ExitCode.ERROR


@contextmanager
def failures(session: Session) -> Iterator[None]:
    """Turn an exception into one message + a hint + an exit code (a traceback only with --debug)."""
    try:
        yield
    except (typer.Exit, typer.Abort, *CLICK_ERRORS):  # usage errors keep exit code 2
        raise
    except BaseException as e:  # KeyboardInterrupt included: what finished is saved, say how to go on
        if session.debug or isinstance(e, SystemExit):
            raise
        message, hint, code = explain(e, session.run_id)
        session.ui.error(message, hint)
        raise typer.Exit(int(code)) from None


# --- shared checks ---------------------------------------------------------------------------------------


def _hero_clips(value: int) -> int:
    if value not in (0, 1):
        raise typer.BadParameter(
            "0 or 1 for now. The video target is 6 images + 1 approval-gated hero clip; more clips would need "
            "the director to choose several scenes, and an approval and a price check for each clip.")
    return value


def _not_blank(value: str) -> str:
    if not value.strip():
        raise typer.BadParameter("say in a few words what the video is about")
    return value.strip()


def preflight(real: bool, research: bool = False) -> None:
    """Fail before anything is spent: ffmpeg, the LLM key, fal's key for real media, and the vector
    store when the run will research (a database that is down fails here, before a run folder exists)."""
    from shortsmith import llm

    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise Problem(f"{' and '.join(missing)} not found on PATH",
                      "install ffmpeg (it includes ffprobe), open a new terminal, then `shortsmith doctor`")
    p = llm.provider()
    if not os.environ.get(p.key_env):
        name = os.environ.get("LLM_PROVIDER", "openai")
        raise Problem(f"{p.key_env} is not set (LLM_PROVIDER={name})",
                      "put it in .env in the folder you run from (see .env.example); `shortsmith doctor` checks")
    if real and not os.environ.get("FAL_KEY"):
        raise Problem("FAL_KEY is not set, and --real makes images on fal.ai", "add it to .env, or use --mock")
    if research:
        from shortsmith.research import rag

        rag.default_kb()  # opens the selected store (cached: the researcher node reuses it); StoreUnavailable if down


def set_media(real: bool) -> None:
    """MOCK_MEDIA for this process; the MCP server (--mcp) inherits it."""
    os.environ["MOCK_MEDIA"] = "0" if real else "1"


def confirm_real(session: Session, lines: list[str], yes: bool) -> None:
    """--real spends money: show the estimate, then ask (or take --yes). Never silent."""
    ui = session.ui
    ui.say(Text("--real: images (and an approved hero clip) are made on fal.ai and PAID.", style="bold yellow"))
    for line in lines:
        ui.say(f"  {line}")
    if yes:
        ui.say("  confirmed by --yes")
        return
    answer = ui.confirm("  spend this?")
    if answer is None:
        raise Problem("--real needs a yes, and there is no terminal to ask", "add --yes to confirm the spending",
                      ExitCode.USAGE)
    if not answer:
        raise Problem("cancelled: nothing was made or spent", "run it with --mock (free placeholders)", ExitCode.ERROR)


# --- the global options ------------------------------------------------------------------------------------


def _version(value: bool) -> None:
    if value:
        typer.echo(f"shortsmith {package_version('shortsmith')}")
        raise typer.Exit()


@app.callback()
def main_options(
    ctx: typer.Context,
    debug: Annotated[bool, typer.Option("--debug", envvar="SHORTSMITH_DEBUG",
                                        help="Show the full traceback when something fails.")] = False,
    plain: Annotated[bool, typer.Option("--plain", help="Plain output: no colours, spinners or box lines "
                                        "(automatic when the output is piped or redirected).")] = False,
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                          help="Print the version and exit.")] = False,
) -> None:
    """Shortsmith: AI agents that turn one line of text into a ~30-second vertical short video.

    Start with: shortsmith make "your topic". Every command has its own --help.
    """
    import logging

    logging.getLogger("shortsmith").setLevel(logging.INFO if debug else logging.WARNING)
    ctx.obj = Session(ui=Ui(plain), debug=debug)


# --- make / resume ------------------------------------------------------------------------------------------


def _suggest_budget(s: Any, history: list) -> float:
    """Spent so far + what the rest would cost, rounded up to a whole cent: a budget that would finish."""
    from shortsmith import budget

    need = budget.spent(s.run_id, history) + sum(budget.llm_estimate(n, history).usd for n in budget.nodes_left(s))
    return max(0.01, math.ceil(need * 100) / 100)


def run_graph(session: Session, run_id: str, payload: Any, answer: Answer | None, trace: bool) -> ExitCode:
    """Drive the graph with progress on stderr, then print the outcome on stdout. Returns the exit code."""
    from shortsmith import budget, ledger, runs, tracing
    from shortsmith.agents import graph
    from shortsmith.agents.state import RunState

    ui = session.ui
    if trace:
        with ui.spinner("starting MLflow tracing"):
            tracing.setup()
    progress = NodeProgress(ui, graph.describe)
    before = len(ledger.entries(ledger.DEFAULT_PATH, run_id))
    start = time.perf_counter()
    stopped: budget.BudgetExceeded | None = None

    def ask(question: dict) -> str | None:  # the paused node's spinner must be gone before a prompt is shown
        progress.stop()
        return ui.ask_approval(question, answer.value if answer else None)

    with graph.open_graph() as app_, runs.marked_running(runs.RUNS / run_id),             tracing.span(f"run {run_id}", "AGENT", inputs={"resume": payload is None},
                         tags={"run_id": run_id, "resumed": str(payload is None).lower()}) as root:
        try:
            graph.drive(app_, payload, run_id, ask, on_update=progress.finish, on_start=progress.start)
        except budget.BudgetExceeded as e:  # a clean stop: the checkpoint of the last finished node stays
            stopped = e
        finally:
            progress.stop()
        snapshot = app_.get_state(graph.config(run_id))
        root.set_outputs({"next": list(snapshot.next), "stopped": str(stopped) if stopped else None})
    s = RunState.model_validate(snapshot.values)
    history = ledger.entries(ledger.DEFAULT_PATH)
    lines = [e for e in history if e.run_id == run_id]
    seconds = time.perf_counter() - start

    views.node_table(ui, s, lines, progress.estimates)
    views.money(ui, s, lines, len(lines) - before)
    if root.trace_id:
        views.kv(ui, "trace", f"{root.trace_id} (MLflow: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000)")
    if stopped:
        views.kv(ui, "stopped", Text(f"over budget before {stopped.node} after {seconds:.0f} s", style="bold yellow"))
        views.kv(ui, "", f"spent {usd(stopped.spent)} + {stopped.node} ~{usd(stopped.estimate)} > budget {usd(stopped.limit)} "
                   "(list price; the free tier bills $0). State saved; nothing was sent for that step.")
        views.kv(ui, "next", f"shortsmith resume {run_id} --budget {_suggest_budget(s, history):.2f}")
        return ExitCode.OVER_BUDGET
    if snapshot.next:
        views.kv(ui, "paused", Text(f"at the hero-clip question after {seconds:.0f} s (state saved)", style="bold yellow"))
        views.kv(ui, "next", f"shortsmith resume {run_id} --approve yes   (or --approve no: Ken Burns for that scene)")
        return ExitCode.PAUSED
    views.finished(ui, s, seconds)
    return ExitCode.OK


@app.command(rich_help_panel="Make videos", short_help="Make a video from a topic (writer, director, critic loop).")
def make(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(metavar="TOPIC", help="What the video is about, in a few words.", callback=_not_blank)],
    lang: Annotated[Lang, typer.Option("--lang", "-l", envvar="SHORTSMITH_LANG", help="Language of the video.")] = Lang.en,
    research: Annotated[bool, typer.Option("--research", help="Ground the script in this month's YouTube trends and the hook library (1 more LLM call).")] = False,
    brand: Annotated[str | None, typer.Option(metavar="NAME", help="Write for a brand in data/hooks/brands/, e.g. anor-va-dolchin (implies --research).")] = None,
    approve: Annotated[Answer | None, typer.Option(help="Answer the hero-clip question in advance. Without it you are asked; with no terminal the run pauses.", rich_help_panel="Money")] = None,
    hero_clips: Annotated[int, typer.Option("--hero-clips", metavar="N", callback=_hero_clips, help="AI hero clips: 1 = the director's one approval-gated clip, 0 = Ken Burns only.", rich_help_panel="Money")] = 1,
    budget_usd: Annotated[float | None, typer.Option("--budget", min=0.0, envvar="SHORTSMITH_BUDGET", metavar="USD", help="Hard limit for the whole run in US dollars, LIST price (e.g. 0.05). Checked before every priced step; the run stops cleanly and can be resumed.", rich_help_panel="Money")] = None,
    media: Annotated[bool | None, typer.Option("--mock/--real", envvar="MOCK_MEDIA", help="--mock: free placeholder images and clips (the default). --real: paid images on fal.ai, after a cost estimate and a yes.", show_default=False, rich_help_panel="Money")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm --real without asking (for scripts).", rich_help_panel="Money")] = False,
    max_rounds: Annotated[int, typer.Option(min=0, max=5, metavar="N", help="Critic -> revise rounds at most.", rich_help_panel="Critic loop")] = 2,
    pass_score: Annotated[float, typer.Option(min=0.0, max=10.0, metavar="SCORE", help="Critic score (0-10) at which the video is done.", rich_help_panel="Critic loop")] = 7.0,
    critic_model: Annotated[str | None, typer.Option(envvar="SHORTSMITH_CRITIC_MODEL", metavar="MODEL", help="The critic's model; default gemini-3.8-flash. Each free Flash model allows 20 requests a day, so gemini-3.6-flash is the fallback.", show_default=False, rich_help_panel="Critic loop")] = None,
    prompt: Annotated[PromptVersion, typer.Option(help="Writer prompt version.", rich_help_panel="Advanced")] = PromptVersion.v2,
    mcp: Annotated[bool, typer.Option("--mcp", help="The producer calls the media tools through the MCP server.", rich_help_panel="Advanced")] = False,
    no_trace: Annotated[bool, typer.Option("--no-trace", help="Don't record the run in MLflow (sqlite:///mlflow.db).", rich_help_panel="Advanced")] = False,
    simple: Annotated[bool, typer.Option("--simple", help="The straight Phase 2 pipeline instead: one script, images, voice, video. No critic, no checkpoints.", rich_help_panel="Advanced")] = False,
    music: Annotated[Path | None, typer.Option(exists=True, dir_okay=False, metavar="FILE", help="Your own music file (with --simple); default: a generated bed.", rich_help_panel="Advanced")] = None,
    crash_before: Annotated[str | None, typer.Option(hidden=True, help="Demo: kill the process when this node starts.")] = None,
) -> None:
    """Make a video from a topic: research (optional), writer, director, hero-clip approval, producer,
    editor and a critic loop, saved after every step.

    Example: shortsmith make "coffee shop in Tashkent, cozy autumn vibe" --lang uz --research
    """
    session: Session = ctx.obj
    graph_only = ["approve", "hero_clips", "max_rounds", "pass_score", "prompt", "mcp", "critic_model", "crash_before"]
    given = {name for name in graph_only if getattr(ctx.get_parameter_source(name), 'name', '') == 'COMMANDLINE'}
    if simple and given:
        raise typer.BadParameter(f"{', '.join('--' + g.replace('_', '-') for g in sorted(given))}: only for the graph "
                                 "pipeline; --simple makes one script with no critic or approval step")
    if music and not simple:
        raise typer.BadParameter("--music is for --simple (the graph's editor makes its own music bed)")
    real = media is False
    with failures(session):
        from shortsmith import budget, ledger

        preflight(real, research=research or brand is not None)
        if critic_model:
            from shortsmith.agents import critic

            critic.CRITIC_MODEL = critic_model
        if simple:
            raise typer.Exit(int(_make_simple(session, topic, lang.value, research, brand, music, real, yes, budget_usd)))
        if real:
            est = budget.run_estimate(research or brand is not None, hero_clips, max_rounds, True,
                                      ledger.entries(ledger.DEFAULT_PATH))
            confirm_real(session, views.estimate_lines(est), yes)
        set_media(real)
        from shortsmith.agents import graph

        if crash_before:
            os.environ[graph.CRASH_ENV] = crash_before
        state = graph.new_state(topic, lang.value, brand, research, prompt.value, max_rounds, pass_score, mcp,
                                budget_usd=budget_usd, hero_clips=hero_clips, media="real" if real else "mock")
        session.run_id = state.run_id
        views.header(session.ui, state)
        code = run_graph(session, state.run_id, state.model_dump(), approve, trace=not no_trace)
    raise typer.Exit(int(code))


def _make_simple(session: Session, topic: str, lang: str, research: bool, brand: str | None, music: Path | None,
                 real: bool, yes: bool, budget_usd: float | None) -> ExitCode:
    """--simple: the Phase 2 straight pipeline (make.py's functions), with the ledger tagged by run."""
    from shortsmith import budget, ledger, llm
    from shortsmith import make as straight
    from shortsmith.agents import graph, researcher
    from shortsmith.media import ffmpeg, images
    from shortsmith.models import Script

    ui = session.ui
    history = ledger.entries(ledger.DEFAULT_PATH)
    est = budget.run_estimate(research or brand is not None, 0, 0, real, history,
                              llm_nodes=["researcher"] if research or brand else [])
    est.llm_usd += budget.call_cost("writer", llm.provider().default_model, history)[0]  # one script, one call
    if real:
        confirm_real(session, views.estimate_lines(est), yes)
    if budget_usd is not None and est.first_pass_usd > budget_usd:  # no checkpoints here: check once, up front
        raise Problem(f"over budget: this run would cost ~{usd(est.first_pass_usd)} > budget {usd(budget_usd)} (list price)",
                      "raise --budget, or drop --research / --real", ExitCode.OVER_BUDGET)
    set_media(real)
    run_dir = graph.new_run_id()
    session.run_id = None  # a straight run can't be resumed: no hint to resume it
    token = ledger.CURRENT_RUN.set(run_dir.name)  # tag its ledger lines, so list/show/cost know the run
    start = time.perf_counter()
    try:
        ui.say(Text(f"run {run_dir.name} | {topic!r} ({lang}) | straight pipeline (no critic, no checkpoints) | "
                    f"media: {'REAL (paid)' if real else 'mock placeholders (free)'}", style="bold"))
        found = None
        if research or brand:
            with ui.spinner("researching (YouTube trends + hook library)"):
                found = researcher.research(topic, lang, brand)
        with ui.spinner("writing the script"):
            straight.new_run(topic, lang, found, run_dir=run_dir)
        script = Script.load(run_dir / "script.json")
        with ui.spinner(f"images ({'mock placeholders' if images.mock_media() else 'FLUX, paid'})"):
            straight.make_images(script, run_dir)
        with ui.spinner("voice (edge-tts)"):
            voiceovers = straight.make_voice(script, run_dir)
        with ui.spinner("editing with ffmpeg"):
            final = ffmpeg.assemble(run_dir, script, voiceovers, music)
    finally:
        ledger.CURRENT_RUN.reset(token)
    lines = ledger.entries(ledger.DEFAULT_PATH, run_dir.name)
    ui.print(Text(f"{script.title!r} ({script.language})", style="bold"))
    ui.print(Text(straight.summary(script, voiceovers, final)))
    views.kv(ui, "cost", f"{usd(sum(e.cost_usd for e in lines))} list price | {len(lines)} calls | made in "
                   f"{time.perf_counter() - start:.0f} s")
    views.kv(ui, "next", f"shortsmith show {run_dir.name}")
    return ExitCode.OK


@app.command(rich_help_panel="Make videos", short_help="Continue a stopped or paused run from its checkpoint.")
def resume(
    ctx: typer.Context,
    run: Annotated[str, typer.Argument(metavar="RUN", help="The run: its id, the start of it, or 'latest'.")],
    approve: Annotated[Answer | None, typer.Option(help="Answer a paused hero-clip question.")] = None,
    budget_usd: Annotated[float | None, typer.Option("--budget", min=0.0, metavar="USD", help="A new budget for the run (list price); saved with it.")] = None,
    media: Annotated[bool | None, typer.Option("--mock/--real", help="Change the run's media for the rest of it. Default: what the run started with.", show_default=False)] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm --real without asking.")] = False,
    critic_model: Annotated[str | None, typer.Option(envvar="SHORTSMITH_CRITIC_MODEL", metavar="MODEL", help="The critic's model for this resume.", show_default=False)] = None,
    no_trace: Annotated[bool, typer.Option("--no-trace", help="Don't record this resume in MLflow.")] = False,
) -> None:
    """Continue a run from its last checkpoint: after a crash, a budget stop, or a paused approval.
    Nothing that already finished is paid for again."""
    session: Session = ctx.obj
    with failures(session):
        from shortsmith import budget, ledger, runs
        from shortsmith.agents import graph, producer
        from shortsmith.agents.state import RunState

        folder = runs.resolve(run)
        session.run_id = folder.name
        if (pid := runs.running_pid(folder)) is not None:  # two processes on one checkpoint would both pay
            raise Problem(f"run {folder.name} is running right now (process {pid})",
                          "wait for it to finish, or stop it with Ctrl+C in its window, then resume")
        with graph.open_graph() as app_:
            snapshot = app_.get_state(graph.config(folder.name))
            if not snapshot.values:
                raise Problem(f"run {folder.name} has no checkpoint, so it can't be resumed",
                              "runs made with --simple or before Phase 6 have none; "
                              f"`shortsmith rerun-scene {folder.name} <scene>` works on their files")
            if not snapshot.next:
                session.ui.say(f"run {folder.name} already finished: {snapshot.values.get('final_path')} (nothing to do)")
                raise typer.Exit(int(ExitCode.OK))
            s = RunState.model_validate(snapshot.values)
            real = (s.media == "real") if media is None else media is False
            updates: dict[str, Any] = {}
            if budget_usd is not None and budget_usd != s.budget_usd:
                updates["budget_usd"] = budget_usd
            if real != (s.media == "real"):
                updates["media"] = "real" if real else "mock"
            preflight(real, research="researcher" in budget.nodes_left(s))
            if real:
                images_left = len(producer.pending(s.folder, s.plan, s.hero).images) if s.plan else 6
                est = budget.run_estimate(False, 0 if s.hero is None or s.hero.approved is False else 1,
                                          s.max_rounds, True, ledger.entries(ledger.DEFAULT_PATH),
                                          images_left=images_left, llm_nodes=budget.nodes_left(s))
                confirm_real(session, views.estimate_lines(est), yes)
            if updates:  # saved in the checkpoint as a new state version: later resumes keep it
                app_.update_state(graph.config(folder.name), updates)
                s = s.model_copy(update=updates)
        set_media(real)
        if critic_model:
            from shortsmith.agents import critic

            critic.CRITIC_MODEL = critic_model
        views.header(session.ui, s, resumed=", ".join(snapshot.next))
        code = run_graph(session, folder.name, None, approve, trace=not no_trace)
    raise typer.Exit(int(code))


@app.command("rerun-scene", rich_help_panel="Make videos", short_help="Redo one scene (image and/or voice) and re-cut the video.")
def rerun_scene(
    ctx: typer.Context,
    run: Annotated[str, typer.Argument(metavar="RUN", help="The run: its id, the start of it, or 'latest'.")],
    scene: Annotated[int, typer.Argument(min=1, metavar="SCENE", help="The scene number (shortsmith show lists them).")],
    prompt: Annotated[str | None, typer.Option("--prompt", "-p", metavar="TEXT", help="A new picture prompt (in English). The run's style words are added if it lacks them.")] = None,
    narration: Annotated[str | None, typer.Option("--narration", "-n", metavar="TEXT", help="New spoken words for the scene.")] = None,
    image: Annotated[bool, typer.Option("--image", help="Make the image again even with the same prompt.")] = False,
    voice: Annotated[bool, typer.Option("--voice", help="Record the voice again even with the same words.")] = False,
    approve: Annotated[Answer | None, typer.Option(help="If the hero scene's picture changes: make a new clip (yes) or use Ken Burns (no).")] = None,
    media: Annotated[bool | None, typer.Option("--mock/--real", help="Default: what the run was made with.", show_default=False)] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm --real without asking.")] = False,
) -> None:
    """Redo one scene and cut the video again. Only that scene is made again (nothing else is paid
    for); the replaced files are kept in the run's history/rerun_N/ folder. No LLM call.

    With no option, the scene's image and voice are both made again as they are.
    Example: shortsmith rerun-scene latest 3 --prompt "close-up of hands pouring tea"
    """
    session: Session = ctx.obj
    with failures(session):
        from shortsmith import rerun, runs
        from shortsmith.agents.state import RunState
        from shortsmith.media import animate, images
        from shortsmith.models import Script

        folder = runs.resolve(run)
        snapshot = runs.snapshots([folder.name]).get(folder.name)
        state = RunState.model_validate(snapshot.values) if snapshot is not None and snapshot.values else None
        if snapshot is not None and snapshot.next:  # the graph still has work to do on this folder
            status = runs.status_of(snapshot, runs.running_pid(folder))
            raise Problem(f"run {folder.name} is not finished ({status.label}): a rerun now would be undone by the "
                          "graph's next steps", f"finish it first: shortsmith resume {folder.name}")
        if not (folder / "script.json").exists():
            raise Problem(f"run {folder.name} has no script.json: it never reached the producer",
                          "make a new run: shortsmith make \"topic\"")
        real = media is False if media is not None else (state.media == "real" if state else not images.mock_media())
        preflight(real)
        hero = state.hero if state else None
        req = rerun.RerunRequest(scene=scene, prompt=prompt, narration=narration, image=image, voice=voice)
        preview = rerun.plan_change(Script.load(folder / "script.json"), req, hero)
        if real:
            from shortsmith import ledger

            image_usd = ledger.MEDIA_PRICES[images.FLUX_MODEL] if preview.image else 0.0
            lines = [f"image: {usd(image_usd)}" if preview.image else "image: kept", "voice: free (edge-tts)"]
            if preview.hero_question:
                lines.append(f"hero clip: ~{usd(ledger.MEDIA_PRICES[animate.CLIP_MODEL], 2)} if you say yes to it")
            confirm_real(session, lines, yes)
        set_media(real)
        parts = [p for p, on in (("image", preview.image), ("voice", preview.voice)) if on]
        session.ui.say(Text(f"rerun scene {scene} of {folder.name}: {' and '.join(parts)} | media: "
                            f"{'REAL (paid)' if real else 'mock placeholders (free)'}", style="bold"))
        answer = None
        if preview.hero_question:  # asked before the spinner starts, and before anything is made
            answer = session.ui.ask_approval(preview.hero_question, approve.value if approve else None)
            if answer is None:
                views.kv(session.ui, "paused", Text("nothing was done: the hero-clip question needs an answer", style="bold yellow"))
                views.kv(session.ui, "next", f"add --approve yes (a new clip) or --approve no (Ken Burns for scene {scene})")
                raise typer.Exit(int(ExitCode.PAUSED))
        with session.ui.spinner(f"scene {scene}: making the {' and '.join(parts)}, then cutting the video"):
            record = rerun.rerun_scene(folder, req, hero, lambda question: answer)
    ui = session.ui
    views.kv(ui, "made", f"{record.made} (scene {record.scene}; every other scene untouched)")
    if record.prompt_after != record.prompt_before:
        views.kv(ui, "picture", record.prompt_after)
    if record.narration_after != record.narration_before:
        views.kv(ui, "words", record.narration_after)
    if record.motion_after != record.motion_before:
        views.kv(ui, "motion", f"{record.motion_before} -> {record.motion_after}")
    views.kv(ui, "kept", f"the replaced files and the old final.mp4 in {folder / 'history' / record.label}")
    views.kv(ui, "video", Text(str((folder / "final.mp4").resolve()), style="bold green"))
    views.kv(ui, "cost", f"{usd(record.cost_usd)} list price")


OUTPUT_HELP = ("Write the result to this file (UTF-8) instead of stdout. On Windows PowerShell 5.1, `>` and `|` "
               "re-read our UTF-8 in the console code page (e.g. IBM437) and garble non-English text; this doesn't.")


@contextmanager
def output_to(session: Session, path: Path | None) -> Iterator[Ui]:
    """The Ui a result is printed with: the session's, or a plain one writing UTF-8 into `path`."""
    if path is None:
        yield session.ui
        return
    with path.open("w", encoding="utf-8", newline="\n") as file, redirect_stdout(file):
        yield Ui(plain=True)
    session.ui.say(f"wrote {path.resolve()}")


# --- list / show / cost -------------------------------------------------------------------------------------


@app.command("list", rich_help_panel="Look at runs", short_help="The runs, newest first, with status and cost.")
def list_(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-n", min=0, metavar="N", help="How many runs, newest first; 0 = all.")] = 20,
    as_json: Annotated[bool, typer.Option("--json", help="JSON on stdout, for scripts.")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", metavar="FILE", dir_okay=False, help=OUTPUT_HELP)] = None,
) -> None:
    """The runs, newest first: title, language, critic score, length, cost and where each one stands."""
    session: Session = ctx.obj
    with failures(session):
        from shortsmith import runs

        rows = runs.list_runs(limit or None)
        if not rows and not as_json:
            session.ui.say("no runs yet: shortsmith make \"your topic\"")
            return
        with output_to(session, output) as ui:
            if as_json:
                ui.json([r.model_dump(mode="json") for r in rows])
                return
            table = ui.table("id", "title", "lang", "score", "length", "cost", "status", right=("score", "length", "cost"))
            for r in rows:
                table.add_row(r.id, r.title or "-", r.language or "-", f"{r.score:g}" if r.score is not None else "-",
                              f"{r.length_s:.1f} s" if r.length_s else "-", usd(r.cost_usd), views.status_text(ui, r.status))
            ui.print(table)
        ui = session.ui
        total = len(runs.run_folders())
        more = f"; {total - len(rows)} older: --limit 0" if total > len(rows) else ""
        ui.say(f"{len(rows)} of {total} runs{more}. Cost = list price of the run's ledger lines "
               "(billed $0 on the Gemini free tier; '-' = made before runs were tagged). Details: shortsmith show <id>")


@app.command(rich_help_panel="Look at runs", short_help="One run in detail: script, critic, cost, files.")
def show(
    ctx: typer.Context,
    run: Annotated[str, typer.Argument(metavar="RUN", help="The run: its id, the start of it, or 'latest'.")] = "latest",
    as_json: Annotated[bool, typer.Option("--json", help="JSON on stdout, for scripts.")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", metavar="FILE", dir_okay=False, help=OUTPUT_HELP)] = None,
) -> None:
    """One run in detail: the script per scene, critic rounds, revisions, reruns, cost per node,
    the files (where final.mp4 is) and the MLflow trace ids."""
    session: Session = ctx.obj
    with failures(session):
        from shortsmith import runs

        folder = runs.resolve(run)
        d = runs.detail(folder, runs.snapshots([folder.name]).get(folder.name))
        with output_to(session, output) as ui:
            if as_json:
                ui.json(d.model_dump(mode="json"))
            else:
                views.show_run(ui, d)


@app.command(rich_help_panel="Look at runs", short_help="Spending by day, run and model; free-tier usage today.")
def cost(
    ctx: typer.Context,
    days: Annotated[int | None, typer.Option(min=1, metavar="N", help="Only the last N days (today included).")] = None,
    run: Annotated[str | None, typer.Option(metavar="ID", help="Only one run (its id, the start of it, or 'latest').")] = None,
    top: Annotated[int, typer.Option(min=1, metavar="N", help="Rows per table.")] = 12,
    as_json: Annotated[bool, typer.Option("--json", help="JSON on stdout, for scripts.")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o", metavar="FILE", dir_okay=False, help=OUTPUT_HELP)] = None,
) -> None:
    """What was spent, from the ledger: by day, by run, by model and step; list price next to what
    was billed (the Gemini free tier bills $0); and today's free-tier usage from our own logs."""
    session: Session = ctx.obj
    with failures(session):
        from shortsmith import costs, ledger, runs
        from shortsmith.agents import critic

        run_id = runs.resolve(run).name if run else None
        report = costs.report(ledger.entries(ledger.DEFAULT_PATH), days, run_id, critic_model=critic.CRITIC_MODEL)
        with output_to(session, output) as ui:
            if as_json:
                ui.json(report.model_dump(mode="json"))
            else:
                views.cost_report(ui, report, top)


# --- doctor / mcp -----------------------------------------------------------------------------------------------


KEYS = {  # what each key is for; doctor says only whether it is set, never any part of it
    "GEMINI_API_KEY": "LLM calls and embeddings (LLM_PROVIDER=gemini)",
    "OPENAI_API_KEY": "LLM calls (LLM_PROVIDER=openai)",
    "FAL_KEY": "real images and hero clips (--real)",
    "YOUTUBE_API_KEY": "trend research (--research)",
}


def store_rows() -> list[tuple[str, bool | None, str]]:
    """doctor's lines about the RAG vector store. The database is a FAIL when it is the selected store and
    can't be reached: --research and retrieve_hooks would fail, and nothing falls back to another store."""
    from shortsmith import llm
    from shortsmith.research import rag, stores

    try:
        s = stores.status(rag.collection_name(llm.provider().embedding_model, llm.EMBED_DIMENSIONS))
    except ValueError as e:  # VECTOR_STORE or LLM_PROVIDER has a value we don't know
        return [("vector store", False, str(e))]
    chosen = f"VECTOR_STORE={os.environ['VECTOR_STORE']}" if os.environ.get("VECTOR_STORE") else "the default"
    if s.backend == "chroma":
        where = "exists" if stores.CHROMA_DIR.exists() else "not created yet: the first --research run makes it"
        return [("vector store", None, f"chroma ({chosen}): collection {s.table} in {stores.CHROMA_DIR}/ ({where})")]
    rows: list[tuple[str, bool | None, str]] = [
        ("vector store", None, f"pgvector ({chosen}): table {s.table} at {s.url or 'DATABASE_URL not set'}"),
        ("Docker", s.docker_running or None, s.docker or ""),
    ]
    if not s.reachable:
        return rows + [("database", False, f"{s.error}; next: {s.hint}")]
    if s.rows is None:
        size = "table not created yet: the first --research run makes it and embeds the chunks (python -m shortsmith.research.migrate copies Chroma's instead, 0 requests)"
    else:
        size = f"{s.rows} rows in {s.table}"
    return rows + [("database", True, f"PostgreSQL {s.postgres}, pgvector {s.pgvector or 'not created yet'}; {size}; {s.seconds} s")]


@app.command(rich_help_panel="Setup", short_help="Check ffmpeg, keys (set or not), media mode and stores.")
def doctor(ctx: typer.Context) -> None:
    """Check the setup without spending anything: ffmpeg, which keys are set (never their values),
    the LLM provider, mock or real media, and the stores (ledger, checkpoints, MLflow, the vector
    store: Docker, the database, pgvector, its rows; the password is never shown)."""
    session: Session = ctx.obj
    with failures(session):
        import importlib.util

        from shortsmith import ledger, llm, runs
        from shortsmith.agents import critic

        rows: list[tuple[str, bool | None, str]] = []  # (check, ok? None = information, detail)
        for tool in ("ffmpeg", "ffprobe"):
            path = shutil.which(tool)
            first = ""
            if path:
                done = subprocess.run([tool, "-version"], capture_output=True, text=True, errors="replace")
                first = done.stdout.splitlines()[0].split(" Copyright")[0] if done.stdout else ""
            rows.append((tool, path is not None, f"{first}; {path}" if path else "not on PATH: install ffmpeg"))
        rows.append((".env", None, "found in this folder (loaded; not printed)" if Path(".env").exists()
                     else "none in this folder (settings come from the environment only)"))
        name = os.environ.get("LLM_PROVIDER", "openai")
        try:
            p = llm.provider()
            rows.append(("LLM provider", True, f"{name}: {p.default_model}; critic {critic.CRITIC_MODEL}"))
            needed = p.key_env
        except ValueError as e:
            rows.append(("LLM provider", False, str(e)))
            needed = ""
        for key, use in KEYS.items():
            is_set = bool(os.environ.get(key))
            rows.append((key, is_set if key == needed else None, f"{'set' if is_set else 'not set'}; {use}"))
        try:
            ledger.cost_usd(critic.CRITIC_MODEL, 0, 0)
            rows.append(("critic price", True, f"{critic.CRITIC_MODEL} has a list price in ledger.py"))
        except ValueError as e:
            rows.append(("critic price", False, str(e)))
        mock = os.environ.get("MOCK_MEDIA", "1") != "0"
        rows.append(("MOCK_MEDIA", None, "mock: free placeholders" if mock else "0: REAL, PAID media for every run"))
        rows.append(("billing", None, f"GEMINI_TIER={os.environ.get('GEMINI_TIER', 'free')} (what `cost` counts as billed)"))
        lines = ledger.entries(ledger.DEFAULT_PATH)
        rows.append(("ledger", None, f"{ledger.DEFAULT_PATH}: {len(lines)} lines, {usd(sum(e.cost_usd for e in lines))} list price"))
        threads = runs.checkpoint_threads()
        rows.append(("checkpoints", None, f"{runs.CHECKPOINTS}: {len(threads)} runs" if runs.CHECKPOINTS.exists() else
                     f"{runs.CHECKPOINTS}: not created yet (the first graph run makes it)"))
        rows.append(("runs", None, f"{runs.RUNS}/: {len(runs.run_folders(threads=threads))} runs"))
        has_mlflow = importlib.util.find_spec("mlflow") is not None
        rows.append(("MLflow", has_mlflow, f"{'installed' if has_mlflow else 'not installed (runs go untraced)'}; "
                     f"{runs.MLFLOW_DB}: {f'{runs.MLFLOW_DB.stat().st_size / 1e6:.1f} MB' if runs.MLFLOW_DB.exists() else 'not created yet'}"))
        rows += store_rows()

        ui = session.ui
        table = ui.table("check", "", "detail")
        for check, ok, detail in rows:
            mark = {True: ("ok", "green"), False: ("FAIL", "bold red"), None: ("-", "dim")}[ok]
            table.add_row(check, Text(mark[0], style=mark[1] if ui.rich_out else ""), detail)
        ui.print(table)
        failed = [check for check, ok, _ in rows if ok is False]
        if failed:
            ui.error(f"not ready: {', '.join(failed)}", "fix the FAIL lines above (.env.example lists the keys)")
            raise typer.Exit(int(ExitCode.ERROR))
        ui.say("ready: `shortsmith make \"your topic\"` costs $0 with mock media on the Gemini free tier")


@app.command(rich_help_panel="Setup", short_help="Start the MCP server on stdio (for Claude Desktop).")
def mcp(
    root: Annotated[Path | None, typer.Option(metavar="FOLDER", help="The folder that holds runs/ (default: this repo).")] = None,
) -> None:
    """Start Shortsmith's MCP server on stdio, as Claude Desktop does (the same as
    python -m shortsmith.mcp_server). It prints only the protocol on stdout; its log goes to stderr."""
    import logging

    from shortsmith import mcp_server

    cli_logger = logging.getLogger("shortsmith")  # the server configures its own log (INFO lines on stderr):
    cli_logger.setLevel(logging.NOTSET)  # give it back the logger as `python -m shortsmith.mcp_server` finds it
    for handler in list(cli_logger.handlers):
        cli_logger.removeHandler(handler)
    mcp_server.main(["--root", str(root)] if root else [])


# --- the entry point ---------------------------------------------------------------------------------------------

GLOBAL_FLAGS = ("--debug", "--plain")


def _move_global_flags(argv: list[str]) -> list[str]:
    """`shortsmith make "x" --debug` works like `shortsmith --debug make "x"`: the hint every error
    prints says "run again with --debug", and people add it at the end."""
    cut = argv.index("--") if "--" in argv else len(argv)
    flags = [a for a in argv[:cut] if a in GLOBAL_FLAGS]
    return flags + [a for a in argv[:cut] if a not in GLOBAL_FLAGS] + argv[cut:]


class _CurrentStderr:
    """A logging stream that is whatever sys.stderr is at that moment (Rich's spinner swaps it)."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def main() -> None:
    """The `shortsmith` command ([project.scripts]) and `python -m shortsmith`."""
    import logging

    from dotenv import load_dotenv

    for stream in (sys.stdout, sys.stderr):  # a redirected stream on Windows is cp1252 otherwise: Uzbek text breaks
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(Path.cwd() / ".env", override=False)  # the real environment wins over .env; flags win over both
    handler = logging.StreamHandler(_CurrentStderr())  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger("shortsmith").addHandler(handler)
    if not sys.stdout.isatty():
        app.rich_markup_mode = None  # piped --help: Click's plain text instead of Rich's boxes
    app(args=_move_global_flags(sys.argv[1:]), prog_name="shortsmith")
