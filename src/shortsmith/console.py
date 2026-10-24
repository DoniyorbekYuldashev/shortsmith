"""How the CLI talks to a person (Step 7.3): Rich in a terminal, plain lines everywhere else.

- stdout carries RESULTS (the summary table, list/show/cost, --json); stderr carries PROGRESS and
  messages. So `shortsmith list --json > runs.json` gets clean JSON, and `shortsmith make ... > out.txt`
  still shows the progress on screen while the summary goes to the file.
- A terminal gets colours, a spinner with the elapsed time while a node runs (a free-tier call can take
  minutes), and tables with box lines. Output that is piped or redirected, TERM=dumb, or --plain gets
  plain lines: no ANSI codes, no animation, and tables as Markdown (ASCII borders, and they read well
  when pasted into notes). NO_COLOR turns colours off and keeps the rest (Rich reads it itself).
- Markup is off everywhere: run text (a narration, "[1, 3]", "[en|uz|ru]") is printed as it is.
"""

import json
import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

PLAIN_WIDTH = 200  # piped output has no terminal width: wide enough that table rows don't wrap

# What a node is doing while the spinner turns (a free-tier critic call can take 6 minutes).
DOING = {
    "researcher": "reading YouTube trends and the hook library",
    "writer": "writing 3 script options, then picking one",
    "director": "planning the shots and the hero scene",
    "approve_hero": "checking whether a hero clip needs a yes",
    "producer": "making the images and voices that are missing",
    "editor": "cutting the video with ffmpeg",
    "critic": "watching 4 frames and judging the video",
    "revise": "rewriting the weak scenes",
}

STATUS_STYLE = {"finished": "green", "running": "cyan", "paused": "yellow", "over_budget": "yellow", "crashed": "red", "no_checkpoint": "dim"}


def usd(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return "$0" if value == 0 else f"${value:.{digits}f}"


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # a replaced or closed stream
        return False


def stdin_interactive() -> bool:
    """Is there a person to ask? (Windows' NUL device also says yes: the EOFError path covers it.)"""
    return sys.stdin is not None and sys.stdin.isatty()


class Ui:
    """The two consoles and the choices made once per command: rich or plain, per stream."""

    def __init__(self, plain: bool = False):
        options: dict[str, Any] = {"highlight": False, "markup": False, "emoji": False}
        if plain:
            options.update(no_color=True, force_terminal=False, force_interactive=False)
        self.out = Console(**options)
        self.err = Console(stderr=True, **options)
        self.rich_out = self.out.is_terminal and not plain and not self.out.is_dumb_terminal
        self.rich_err = self.err.is_terminal and not plain and not self.err.is_dumb_terminal
        for console, stream in ((self.out, sys.stdout), (self.err, sys.stderr)):
            if console is self.out and self.rich_out or console is self.err and self.rich_err:
                continue
            # --plain on a real screen keeps its width; a file or a pipe gets one wide enough not to wrap
            console.width = shutil.get_terminal_size().columns if _isatty(stream) else PLAIN_WIDTH

    # --- messages (stderr) ---

    def say(self, text: str | Text = "", style: str | None = None) -> None:
        self.err.print(text, style=style, soft_wrap=not self.rich_err)

    def warn(self, text: str) -> None:
        self.err.print(Text.assemble(("warning: ", "bold yellow"), text), soft_wrap=not self.rich_err)

    def error(self, message: str, hint: str = "") -> None:
        self.err.print(Text.assemble(("error: ", "bold red"), message), soft_wrap=not self.rich_err)
        if hint:
            self.err.print(Text.assemble(("next: ", "bold"), hint), soft_wrap=not self.rich_err)

    @contextmanager
    def spinner(self, text: str) -> Iterator[None]:
        """A spinner on stderr for a slow step that isn't a node (MLflow's first start takes seconds)."""
        if self.rich_err:
            with self.err.status(text):
                yield
        else:
            self.say(f"{text}...")
            yield

    # --- results (stdout) ---

    def table(self, *columns: str, title: str | None = None, right: tuple[str, ...] = ()) -> Table:
        """A table for stdout: light box lines in a terminal, Markdown when piped."""
        table = Table(title=title, box=box.SIMPLE_HEAD if self.rich_out else box.MARKDOWN,
                      title_justify="left", show_edge=not self.rich_out, header_style="bold" if self.rich_out else "")
        for column in columns:
            table.add_column(column, justify="right" if column in right else "left", overflow="fold")
        return table

    def print(self, *things: Any) -> None:
        if self.rich_out:
            self.out.print(*things)
            return
        with self.out.capture() as captured:  # Rich draws a Markdown table's top and bottom edge as spaces:
            self.out.print(*things, soft_wrap=True)  # drop those lines, and trailing spaces, so a file stays clean
        text = "\n".join(line.rstrip() for line in captured.get().split("\n") if not (line and not line.strip()))
        sys.stdout.write(text)
        sys.stdout.flush()

    def json(self, data: Any) -> None:
        """Clean JSON on stdout, ASCII-only (\\u escapes): survives any console code page or pipe."""
        sys.stdout.write(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
        sys.stdout.flush()

    # --- questions ---

    def confirm(self, question: str) -> bool | None:
        """Yes/no from the person; None when there is nobody to ask (no terminal, or stdin closed)."""
        if not stdin_interactive():
            return None
        try:
            return Confirm.ask(question, console=self.err, default=False)
        except EOFError:
            return None

    def ask_approval(self, question: dict[str, Any], answer: str | None) -> str | None:
        """The hero-clip question, clearly. --approve answers it; else ask; else None (the run pauses)."""
        self.say()
        self.say(Text.assemble(("hero clip? ", "bold yellow"), question["question"]))
        if answer is not None:
            self.say(f"  answered by --approve: {answer}")
            return answer
        reply = self.confirm("  make the clip?")
        if reply is None:
            self.say("  nobody to ask here (no terminal): the run is saved and paused at this question.")
            return None
        return "yes" if reply else "no"


class NodeProgress:
    """make / resume: a spinner while a node runs, one line when it is done (on stderr)."""

    def __init__(self, ui: Ui, describe: Callable[[str, dict], str]):
        self.ui = ui
        self.describe = describe
        self.live: Progress | None = None
        self.estimates: dict[str, float] = {}  # per node, summed when a node runs again (critic, producer)

    def start(self, name: str, estimate: Any) -> None:
        self.stop()
        self.estimates[name] = self.estimates.get(name, 0.0) + estimate.usd
        text = f"{name}: {DOING.get(name, 'running')}" + (f" (est. {usd(estimate.usd)})" if estimate.usd else "")
        if self.ui.rich_err:
            self.live = Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
                                 console=self.ui.err, transient=True)
            self.live.start()
            self.live.add_task(text, total=None)
        else:
            self.ui.say(f"  ... {text}")

    def finish(self, name: str, update: dict) -> None:
        self.stop()
        usage = update.get("usage", {}).get(name)
        calls = getattr(usage, "calls", 0) if usage is not None else 0
        seconds = getattr(usage, "seconds", 0.0) if usage is not None else 0.0
        tokens = (f"{usage.prompt_tokens:>6,} in {usage.completion_tokens:>6,} out  {usd(usage.cost_usd):>8}"
                  if calls else f"{'':>28}")
        line = f"{name:<13}{seconds:>7.1f} s  {tokens}  {self.describe(name, update)}"
        if self.ui.rich_err:
            self.ui.err.print(Text.assemble(("done ", "green"), line))
        else:
            self.ui.say(f"  done {line}")

    def stop(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None
