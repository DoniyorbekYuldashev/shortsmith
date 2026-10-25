# Phase 7 — The CLI: a tool a person can use without reading the code

> Read at your own pace, ask anything. Parts 0–12 are the concepts. Parts 13–17 are what was really built and
> run (from `docs/lessons/phase-7-build-notes.md`, commit 04cfd7e): our commands, real terminal output, the
> budget stop, errors, tests.

---

## 0. The problem: it works, but only for its author

After Phase 6, making a video looks like this:
```bash
uv run --env-file .env python -m shortsmith.make "coffee shop in Tashkent" --graph --research --approve yes
```
To use it you must know a **module path**, which flags exist, what `--graph` means, and where the result went. To
see what a run cost you open `runs/ledger.jsonl`. To redo one scene you write Python. That's a developer's
tool, not a product.

Phase 7's goal, from the plan: **"a person can use the tool without reading the code."**
```bash
shortsmith make "coffee shop in Tashkent" --lang en --budget 0.50
shortsmith list
shortsmith show 2026-10-24_001351
shortsmith rerun-scene 2026-10-24_001351 3 --voice
shortsmith cost
```
Nothing new happens *inside*. The agents, the graph and the media are all done. Phase 7 is **interface
design**: naming, defaults, help, errors, output, and guard rails around money.

---

## 1. The anatomy of a command line

```
shortsmith  rerun-scene  2026-10-24_001351  3  --voice  --narration "New line."  --debug
└─program─┘ └─command──┘ └──argument──────┘ └┘ └flag─┘  └────option + value────┘  └flag─┘
```
- **Program:** what you run (`shortsmith`, `git`, `uv`).
- **Command (subcommand):** a verb, the action (`make`, `list`, `show`). `git commit` and `uv add` work the same way.
- **Arguments (positional):** required values, identified by their **position** (`<run>`, then `<idx>`).
- **Options:** named, usually optional, with a value: `--lang uz`, `--budget 0.50`. Conventions:
  `--long-name`, sometimes a short form `-l`, and `--name=value` also works.
- **Flags:** options with no value, just on or off: `--json`, `--debug`, `--yes`.
- **`--help`** on every level: `shortsmith --help`, `shortsmith make --help`. This is the main way a new user
  learns the tool, so the help text *is* the documentation.

**A rule of thumb:** what the command always needs is an **argument**. What has a sensible default is an
**option**.

---

## 2. How a Python function becomes a command you can type

**`[project.scripts]` in `pyproject.toml`** (an "entry point"):
```toml
[project.scripts]
shortsmith = "shortsmith.cli:main"
```
- When the project is installed into `.venv` (uv does this on `uv sync` / `uv run`), the installer creates a
  small launcher, **`.venv\Scripts\shortsmith.exe`**, that imports `shortsmith.cli` and calls `main()`.
  (Before this phase the entry pointed at `shortsmith:main`, the "hello" stub `uv init` made in Phase 0.)
- Our `main()` does a few things *before* Typer reads your words: loads `.env`, makes stdout/stderr UTF-8, and
  moves `--debug`/`--plain` to the front if you typed them at the end. Then it calls the Typer app.
- **`uv run shortsmith …`** finds that launcher inside the project's `.venv`.
- **`uv tool install -e .`** would put `shortsmith` on your PATH everywhere, like `git`.

`python -m shortsmith.make` (what we used so far) runs a **module** as a script. It works, but you must know
the module's path. An entry point gives the tool **a name**. We keep the old `python -m` path working, because
earlier lessons and the Claude Desktop config use module paths.

---

## 3. argparse vs Click vs Typer

All three turn `sys.argv` (the list of words you typed) into Python values:
- **argparse** (built into Python, used by `make.py` so far): you describe every argument by hand with
  `parser.add_argument(...)`.
- **Click** (a library): commands are decorated functions. Mature, used by Flask and many tools.
- **Typer** (built on Click, by the author of FastAPI): **the CLI is read from the function's type hints**, the
  same idea as Pydantic models becoming JSON schemas, and Phase 3's tool schemas.

```python
from typing import Annotated
import typer

app = typer.Typer(help="Make ~30-second vertical videos from one line of text.")

@app.command()
def make(
    topic: Annotated[str, typer.Argument(help="What the video is about.")],
    lang: Annotated[str, typer.Option(help="en, uz or ru.")] = "en",
    budget: Annotated[float | None, typer.Option(help="Stop before the run's list price goes over this (USD).")] = None,
) -> None:
    """Make a video about TOPIC."""          # this docstring becomes the command's --help text
    ...

@app.command("list")
def list_runs(json_: Annotated[bool, typer.Option("--json", help="Print JSON for scripts.")] = False) -> None:
    """List your runs, newest first."""
    ...
```
- `topic: str` with no default → a **required argument**. `lang: str = "en"` → an **option with a default**.
- `float | None` → Typer checks the type: `--budget abc` is rejected with a clear message before your code runs.
- The **docstring and `help=` texts become `--help`**, so the documentation lives next to the code and can't
  drift away from it.

---

## 4. Exit codes: how a program says "it worked" to another program

Every process ends with a number, its **exit code**:
- **0 = success.** Anything else = some kind of failure.
- In PowerShell you can see it in `$LASTEXITCODE`, in bash in `$?`.
- Scripts, CI pipelines and other programs **decide what to do next from it**: `shortsmith make "x" && echo done`
  runs `echo` only after a 0.

So exit codes are part of a CLI's interface, like its flags. The plan for ours:

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | an error (a bug, a network failure, a missing file) |
| 2 | wrong usage (unknown flag, missing argument): Typer/Click's convention |
| 3 | paused, waiting for your approval (the hero clip) |
| 4 | stopped by the budget |
| 130 | interrupted with Ctrl+C (the Unix convention: 128 + the signal's number, 2) |

3 and 4 aren't errors: the run is **saved and resumable**. But a script calling `shortsmith` must be able to
tell "finished" from "waiting for a human" from "stopped to protect money". That's what distinct codes give it.
(These are the codes the build settled on; they're printed at the bottom of `shortsmith --help`.)

---

## 5. stdout vs stderr in a CLI: the MCP lesson again

You learned it in Phase 4: **stdout is a data channel, stderr is for humans and logs.** The same rule makes a
CLI composable:
```bash
shortsmith list --json > runs.json
```
- the **result** (the JSON) goes to **stdout**, so `runs.json` contains clean JSON;
- **progress, spinners and warnings** go to **stderr**, so you still see them on screen, and they never get
  into the file.

If progress were printed to stdout, `runs.json` would contain "⠋ writer…" lines, and any program reading it
would fail, exactly like the stray `print()` that breaks an MCP connection.

---

## 6. Is anyone watching? TTY detection, ANSI codes, NO_COLOR

A **TTY** (terminal) is an interactive screen. A program can ask whether its stdout is one:
`sys.stdout.isatty()`.
- In your PowerShell window → `True`.
- Piped (`| findstr`), redirected (`> file.txt`), or run by another program (CI, our tests, Phase 8's web
  server) → `False`.

Why it matters:
- **Colours and spinners are made of ANSI escape codes**: invisible instructions inside the text, like
  `\x1b[32m` ("green from here") and `\x1b[0m` ("back to normal"). A terminal obeys them. A file just stores
  them as junk: `[32mdone[0m`.
- **A spinner is an illusion:** the program writes a frame, then `\r` (back to the start of the line), then the
  next frame over it, ~10 times a second. In a file, that's hundreds of lines of garbage.
- **A question needs someone to answer it.** "Continue? [y/N]" with nobody at the keyboard either hangs forever
  or crashes (Phase 6's `EOFError`, E10).

So a good CLI has **two modes**: rich for a human at a terminal, **plain** when not a terminal. Plus the
**`NO_COLOR`** convention: if that environment variable is set, no colours, even in a terminal.

---

## 7. Rich: progress, tables and live output

**Rich** is a Python library for terminal output:
- **`console.status("writer…")`**: a spinner while a slow step runs. Important for us: a free-tier call can take
  minutes, and a silent terminal looks frozen.
- **Tables** (`rich.table.Table`): `list`, `show` and `cost` as aligned columns with headers.
- **Styled text** (`[bold green]done[/]`), panels and markdown.
- **It detects the terminal itself:** when output isn't a TTY, Rich drops colours and animations automatically.
  `Console(stderr=True)` sends everything to stderr (section 5).

```python
from rich.console import Console
from rich.table import Table

err = Console(stderr=True)                    # progress and messages → stderr
with err.status("writer: 3 options…"):
    options = write_options(...)
err.print("[green]✓[/] writer  18.3 s  1,486 in  2,183 out  $0.0059")

table = Table("run", "title", "score", "cost")
table.add_row("2026-10-24_001351", "Campus ID Goldmine", "8.0", "$0.0116")
Console().print(table)                        # the result → stdout
```

---

## 8. Configuration precedence: who wins?

The same setting can come from several places. The usual order, **most specific wins**:
```
1. a flag on this command          shortsmith make "x" --mock
2. an environment variable         $env:MOCK_MEDIA = "1"
3. the .env file                   MOCK_MEDIA=1
4. the program's default           mock images
```
- A flag is about **this one run**; the environment is about **this machine/session**; the default is the
  **safe choice** when nobody said anything.
- This is the "config in the environment" idea (the twelve-factor app): the same code runs on your PC, in tests
  and on a server, and only the environment differs.
- **Defaults must be the safe ones.** Ours: mock media, the free tier, approval before any paid clip. Spending
  money must always be something you asked for.

---

## 9. Guard rails around money

The plan's flags are mostly about **cost control**:
- **`--budget 0.50`**: a hard limit per run.
  - **Before** each priced step: estimate its cost (e.g. from the prompt's tokens and the expected answer size);
    if *spent so far + estimate* would go over, **stop cleanly**: state saved, resumable, exit code 4.
  - **After** each step: the ledger has the real cost, so the next estimate starts from real numbers.
  - Honest limit: an estimate can be wrong (answers vary in length, thinking tokens are unknown in advance).
    A budget guard is "stop *before* the step that would probably cross the line", not a guarantee to the cent.
  - On the free tier everything is **billed $0**. The budget counts **list price**, so it still means something
    and works unchanged on a paid key.
- **`--real`**: real fal images and clips. It must show the estimated cost and ask, or require **`--yes`** in a
  script. Never silently.
- **`--hero-clips 0|1`**: the one expensive piece, off or on, still behind the approval step.
- **Fail before you pay** (Phase 1's rule) applied to the interface: validate the topic, the language, the brand
  and the budget *before* the first paid call.

---

## 10. Designing for humans first, scripts second

A few principles from good CLIs (`git`, `uv`, `gh`), and from the clig.dev guidelines:
- **Human-first output by default:** short, aligned, coloured when it helps. **`--json` for machines:** the same
  data, stable field names, nothing else on stdout. Phase 8's web app can call these commands or reuse their
  functions.
- **Errors say what happened and what to do next**, in one or two lines:
  ```
  error: no run "2026-10-24_0013" (did you mean 2026-10-24_001351?)
  try: shortsmith list
  ```
  not a 40-line Python traceback. **`--debug`** shows the traceback when *you* need it.
- **Suggest the next command:** after a pause, print the exact `shortsmith resume … --approve yes|no` line.
- **Don't ask when nobody can answer:** in non-interactive mode, take the answer from a flag (`--yes`,
  `--approve`), or stop with a clear message and exit code.
- **Consistent names:** a run is always "run" and always the folder id; every command that takes a run accepts
  the same id.
- **Make the safe thing easy and the dangerous thing explicit** (section 9).

---

## 11. Commands that build on saved state

Two of our commands aren't "start from zero":
- **`resume <run>`** continues a paused or crashed run from its last checkpoint (Phase 6). The CLI only has to
  find the run and pass `--approve` if the pause is the hero-clip question.
- **`rerun-scene <run> <idx>`** redoes one scene: the image, the voice, or both, maybe with a new prompt or line,
  then re-renders. It reuses the producer's **manifest** (`media.json`), so it pays only for that scene, and the
  old files go to `history/` like a Critic round.

Both are possible only because earlier phases kept state **on disk** (folders, manifest, checkpoints, ledger).
A CLI is a thin layer over that state; it doesn't own any of its own.

---

## 12. Testing a CLI

Typer (via Click) ships a **`CliRunner`**: it calls your app **in-process**, with arguments as a list, and
captures stdout, stderr and the exit code:
```python
from typer.testing import CliRunner
from shortsmith.cli import app

def test_list_json_is_clean():
    result = CliRunner().invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    json.loads(result.stdout)          # stdout must be valid JSON, nothing else
```
- No real terminal is involved, so this also tests the **plain mode** (no colours, no spinners).
- Fixture run folders and a fixture ledger make `list`, `show` and `cost` testable offline; a fake LLM makes
  `make` and the budget stop testable at $0.

---

## 13. Our commands

**Installed:** typer 0.27.3 and rich 15.0.0. Both were already in `uv.lock`, because `mcp[cli]` uses them, so
there was no download; they're now direct dependencies.

### 13.1 The command tree
```
shortsmith [--debug] [--plain] [--version]
├─ Make videos
│  ├─ make TOPIC        [--lang en|uz|ru] [--research] [--brand NAME] [--approve yes|no] [--hero-clips 0|1]
│  │                    [--budget USD] [--mock|--real] [--yes] [--max-rounds N] [--pass-score SCORE]
│  │                    [--critic-model MODEL] [--prompt v1|v2] [--mcp] [--no-trace] [--simple] [--music FILE]
│  ├─ resume RUN        [--approve yes|no] [--budget USD] [--mock|--real] [--yes] [--critic-model MODEL] [--no-trace]
│  └─ rerun-scene RUN SCENE  [--prompt TEXT] [--narration TEXT] [--image] [--voice] [--approve yes|no] [--mock|--real] [--yes]
├─ Look at runs
│  ├─ list              [--limit N] [--json] [-o FILE]
│  ├─ show [RUN]        [--json] [-o FILE]          (RUN defaults to latest)
│  └─ cost              [--days N] [--run ID] [--top N] [--json] [-o FILE]
└─ Setup
   ├─ doctor            checks ffmpeg, keys (set or not, never their values), media mode, the stores
   └─ mcp               starts the MCP server (an alternative command for Claude Desktop)
```
- **`RUN` is forgiving:** a full run id, a unique start of one (`2026-10-23`), `latest`, or a folder path. An
  ambiguous start lists the matches.
- **`make` runs the Phase 6 graph by default.** `--simple` gives the old Phase 2 straight pipeline.
- **`python -m shortsmith.make` still works exactly as before** (D2). The earlier lessons and tests use it, and
  turning it into a wrapper would have changed its output for no gain.

### 13.2 The new files
| File | What it is |
|---|---|
| `cli.py` | the Typer app: 8 commands, flags, `ExitCode`, errors → one message + `next:` hint, `main()` |
| `console.py` | `Ui`: stdout and stderr consoles, rich or plain per stream; `NodeProgress` (spinner + "done" lines); the questions |
| `views.py` | how results look (the make summary, `show`, `cost`): data in, tables out |
| `runs.py` | the runs as data: `RunSummary`, `RunDetail`, status from the checkpoint, `resolve("latest")` |
| `costs.py` | the ledger by day / run / model and step; list price vs billed; today's free-tier usage |
| `budget.py` | estimates per node, `BudgetExceeded`, `check()` |
| `rerun.py` | one scene through the producer's manifest |
| `__main__.py` | so `python -m shortsmith` = the command |

**The split is the lesson:** `cli.py` only parses words and chooses exit codes. *What* a run is lives in
`runs.py`/`costs.py`, *how it looks* in `views.py`. Phase 8's web API can reuse the data functions and the same
Pydantic models (`RunSummary`, `RunDetail`, `CostReport`) without touching the terminal code.

### 13.3 Settings order, and the safe defaults
**flags > environment variables > `.env` in the current folder > defaults.** It's printed in `--help`.
- `main()` loads `.env` with `override=False`, so a real environment variable wins over `.env`, and a flag wins
  over both.
- Since the command loads `.env` itself, **`uv run shortsmith …` works without `--env-file`**. That removes the
  old "space in the path" gotcha.
- The help shows `[env var: …]` and `[default: …]` next to each option. For example, `SHORTSMITH_BUDGET`,
  `SHORTSMITH_LANG` and `MOCK_MEDIA` can all be set once in `.env`.
- **The defaults** (D14):
  - the graph pipeline, because the Phase 6 work is the product;
  - **mock media**: a forgotten setting must never cost money;
  - `--hero-clips 1`, the video target;
  - no budget, but every run shows its cost;
  - `show` shows the latest run.

### 13.4 How `--budget` works (`budget.py`)
- **It lives in the graph, not the CLI** (D6). The Phase 6 node wrapper calls `budget.check()` **before every
  node**, so Phase 8's web API gets the same hard stop for free.
- **spent** = this run's own ledger lines (tagged with its run id), across every process that worked on it.
- **The estimate** for a call is the **most expensive of the last 10 similar calls** (same step, same model). With
  no history it uses a first-call guess from Phase 6's token counts.
  - **Why the maximum and not the average** (D7): a budget is a limit. An estimate that's usually a bit high
    stops a run slightly early; an average-based one would go over about half the time.
  - It also adjusts by itself when a prompt gets longer: the new calls become the history.
- **If `spent + estimate > budget`**, it raises `BudgetExceeded` **before anything is sent**. LangGraph keeps the
  checkpoint of the last finished node. The CLI prints the numbers, suggests a budget that would finish, and exits
  with 4.
- **The budget is saved with the run** (a `RunState` field), so `resume --budget 0.02` changes it and a later plain
  `resume` keeps it.
- **Where it can be wrong, measured:**
  - the writer's estimate was $0.0063, but it really cost $0.0092. All 3 Uzbek options broke the word rule, so it
    wrote again, and the estimate didn't expect a retry;
  - the critic was estimated at $0.0196 and cost $0.0205.

  So a run can pass its budget by **at most one node's error**: the check runs again before the next node.

### 13.5 Running, paused, crashed: how `list` knows
A checkpoint can't tell "a process is working on the next node" from "the process died before finishing it". Both
look like `next = critic` with no error. So the CLI writes `runs/<id>/.running` with its process id while it
drives a run, and removes it at the end. The marker gives `list` and `resume` two answers:
- **process alive:** `list` says "running (critic)", and `resume` refuses to start a second process on the same
  run, which would pay for the critic twice;
- **process gone, marker left behind:** "crashed at critic".

This is bug E5: without the marker, `list` called a live run "crashed".

---

## 14. Real terminal output

### `shortsmith --help` (Rich, in a terminal; shortened)
```
 Usage: shortsmith [OPTIONS] COMMAND [ARGS]...
 Shortsmith: AI agents that turn one line of text into a ~30-second vertical short video.
╭─ Make videos ─────────────────────────────────────────────────────────────╮
│ make         Make a video from a topic (writer, director, critic loop).    │
│ resume       Continue a stopped or paused run from its checkpoint.         │
│ rerun-scene  Redo one scene (image and/or voice) and re-cut the video.     │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ Look at runs ────────────────────────────────────────────────────────────╮
│ list   The runs, newest first, with status and cost.                       │
│ show   One run in detail: script, critic, cost, files.                     │
│ cost   Spending by day, run and model; free-tier usage today.              │
╰────────────────────────────────────────────────────────────────────────────╯
 Settings, highest first: flags > environment variables > .env in the current folder > defaults.
 Exit codes: 0 ok, 1 error, 2 usage, 3 paused for an approval, 4 over budget, 130 interrupted.
 Money: every call is priced at list price in runs/ledger.jsonl; on the Gemini free tier it is billed $0.
```

### A full `make` run (live run 1, Uzbek, with research)
`uv run shortsmith make "tea houses of Samarkand, slow mornings" --lang uz --research --approve yes`

While a node runs, **one spinner line redraws in place** with the elapsed time and the node's estimate:
`⠦ critic: watching 4 frames and judging the video (est. $0.0051) 0:01:09`. When it finishes, the spinner is
replaced by a "done" line:
```
done researcher       8.7 s   1,522 in    298 out   $0.0012  notes: 2 hook patterns, 2 angles
done writer          13.1 s   2,355 in  3,402 out   $0.0092  3 options, picked 1: "Tonggi Samarqandni ko'ring." | rejected: 3
done director         5.0 s     750 in    505 out   $0.0015  hero = scene 4: Steam rises gracefully from the spout ...
hero clip? Scene 4 (...) as a 5-second AI clip costs ~$0.30 with real media; MOCK_MEDIA=1 makes a free placeholder clip now. Continue?
  answered by --approve: yes
done producer        14.7 s                                round 0: images 1-6; voices 1-6; clip 4
done editor          28.8 s                                runs\2026-10-24_182435\final.mp4
done critic         135.4 s   5,379 in    828 out   $0.0071  round 0: score 5.5, weak [1]
done revise           1.7 s     804 in    478 out   $0.0014  scene 1 narration
...
done critic         266.3 s   5,400 in    504 out   $0.0059  round 2: score 6, weak [6]
```
then the summary, **on stdout**:
```
 node           calls   tokens in   tokens out   list price   estimate   seconds
 researcher         2       1,538          298      $0.0012    $0.0014       8.7
 writer             3       2,355        3,402      $0.0092    $0.0063      13.1
 director           1         750          505      $0.0015    $0.0016       5.0
 critic             3      16,172        2,227      $0.0205    $0.0196     567.2
 revise             2       1,626        1,042      $0.0031    $0.0029       7.8
cost     $0.0355 list price | $0 billed (Gemini free tier) | 11 calls | 29,915 tokens
video    D:\All codes\shortsmith\runs\2026-10-24_182435\final.mp4  (27.1 s, 1080x1920, 3.9 MB)
critic   5.5 -> 6 -> 6 | max rounds (2 revisions; last score 6)
done     in 761 s. Details: shortsmith show 2026-10-24_182435 | redo a scene: shortsmith rerun-scene ...
```
- **Progress went to stderr, results to stdout** (§5). `make … > summary.txt` keeps the table in the file and the
  spinner on screen.
- **"rejected: 3"**: the writer's first 3 Uzbek options all broke the word rule, so it wrote again. That's exactly
  the case the estimate missed.
- **The critic waited 135, 165 and 266 s** in the free tier's queue: 567 of the 761 seconds. That's why the
  spinner shows elapsed time; without it, the terminal looked frozen.
- The last line **suggests the next commands**, as §10 recommends.

### `list`, `cost` and `doctor`
```
| id                | title               | lang | score | length |    cost | status                    |
|-------------------|---------------------|------|-------|--------|---------|---------------------------|
| 2026-10-24_184531 | Palaces Underground | en   |   6.2 | 34.4 s | $0.0131 | finished                  |
| 2026-10-24_182435 | Samarqand tongi     | uz   |     6 | 26.7 s | $0.0355 | finished                  |
| 2026-10-24_001351 | Campus ID Goldmine  | en   |     8 | 29.2 s | $0.0116 | finished                  |
```
(Plain mode prints **Markdown tables**: ASCII, readable in a file, and they render in notes.)

`cost` also shows **today's free-tier use, from our own logs**:
```
Gemini today (since midnight Pacific = 07:00 UTC)
| model            | requests | free daily limit | left (by our count) |
| gemini-3.8-flash |        4 |               20 |                  16 |
YouTube  202 units today (our cap 2,000; Google's free quota 10,000)
note     Google can't be asked how much free quota is left: these are the calls OUR logs recorded.
```
This is honest about its limits: Google's real remaining quota can't be queried, and refused (429) requests may
also count.

`doctor` checks the setup **without any network call and without printing a key**:
```
| ffmpeg          | ok | ffmpeg version 9.0.2-full_build ...           |
| GEMINI_API_KEY  | ok | set; LLM calls and embeddings                 |
| MOCK_MEDIA      | -  | mock: free placeholders                       |
| ledger          | -  | runs\ledger.jsonl: 173 lines, $0.2148 list price |
ready: `shortsmith make "your topic"` costs $0 with mock media on the Gemini free tier
```
(A "does the key work?" check would spend one of the day's 20 Flash requests. Checking set or not set catches the
usual mistakes for free, D15.)

### `--json`, for scripts
```json
[{"id": "2026-10-24_184531", "title": "Palaces Underground", "language": "en", "score": 6.2,
  "length_s": 34.4, "cost_usd": 0.01305035, "billed_usd": 0.0,
  "status": {"kind": "finished", "node": null, "label": "finished"}, ...}, ...]
```
It's **ASCII-only on purpose** (D5): Uzbek letters come out as `\u02bb`-style escapes, which any JSON reader turns
back. The reason is the next part.

### Windows: why `>` garbled Uzbek text (E6)
Your terminal is **Windows PowerShell 5.1**, with the console code page IBM437:

| How you ran it | What came out |
|---|---|
| straight to the screen | `— oʻzgacha ☕`: right |
| `shortsmith show … > show.txt` | `ΓÇö o╩╗zgacha Γÿò`: garbled |
| `shortsmith show … -o show.md` | right (Python writes the UTF-8 file itself) |
| `shortsmith show --json \| ConvertFrom-Json` | right (the JSON is ASCII) |

**Why:** PowerShell 5.1 doesn't pass a program's bytes through `>` or `|` unchanged. It **decodes** them with the
console's code page (IBM437, not UTF-8), then **re-encodes** them (as UTF-16 for `>`). Our UTF-8 was correct when
it left Python; it was broken afterwards. That's why `list`, `show` and `cost` got **`-o FILE`**.

The build session's own shell has UTF-8 defaults and looked fine. It was found only by checking the file's first
bytes **in your terminal**. Lesson: **test where the user runs it.**

---

## 15. The budget stop and resume; `--real`'s guard; rerun-scene

### A budget stop, then a resume (live run 2: "why Tashkent metro stations look like palaces")
```
budget: $0.0100 for the whole run, in LIST price (the Gemini free tier bills $0; the budget counts what it would cost on a paid tier)
  done writer          14.2 s   1,162 in  2,064 out   $0.0055  3 options, picked 1: 'Ever rode a subterranean palace?'
  done director         6.0 s     700 in    566 out   $0.0016  hero = scene 1: ...
  done producer        54.8 s                                round 0: images 1-6; voices 1-6; clip 1
  done editor          55.7 s                                runs\2026-10-24_184531\final.mp4
budget   $0.0071 of $0.0100 used (list price)
stopped  over budget before critic after 132 s
         spent $0.0071 + critic ~$0.0074 > budget $0.0100 (list price; the free tier bills $0). State saved; nothing was sent for that step.
next     shortsmith resume 2026-10-24_184531 --budget 0.02
```
That's **exit code 4**, and `list` showed the run as "over budget before critic". Then
`shortsmith resume 2026-10-24_184531 --budget 0.02`:
```
done critic         132.2 s   5,389 in    500 out   $0.0059  round 0: score 6.2, weak [1, 6]
cost     $0.0131 list price | $0 billed | 4 calls (1 in this session)
```
**One new call.** The writer and director weren't paid again: Phase 6's checkpoints at work, now behind a simple
command. (This output is also the **plain-mode** sample: no colours, no spinner, a `...` line when each node
starts, so a log file shows what was running if something hangs.)

### `--real`: never silent
- **It always shows an estimate first:** images, the hero clip (asked again before it's made), revision images
  at most, and the LLM list price.
- **In a terminal** it asks yes/no.
- **With no terminal and no `--yes`** it's a **usage error (exit 2) before any folder is made or any call is sent**:
  `error: --real needs a yes, and there is no terminal to ask` / `next: add --yes to confirm the spending`.
- It checks `FAL_KEY` first. It was **never run for real** in this phase; the tests use fake media.
- A resumed run keeps its own media setting: a mock run never turns real because of the environment.

### `rerun-scene` (live, on run 1)
```
> shortsmith rerun-scene 2026-10-24_182435 2 --prompt "Medium shot of a quiet courtyard teahouse at dawn, a teapot ☕ steaming ..." --narration "Qadimgi hovlida tong salqini — oʻzgacha sokinlik."
made     images 2; voices 2 (scene 2; every other scene untouched)
picture  Medium shot of a quiet courtyard teahouse at dawn, ..., cinematic, photorealistic, 9:16
kept     the replaced files and the old final.mp4 in runs\2026-10-24_182435\history\rerun_1
cost     $0 list price
```
- **Only scene 2's files changed**, which the build checked from the file times. The ledger got no new line,
  because there's no LLM call: you write the words.
- **It adds the run's style words to your new prompt** (", cinematic, photorealistic, 9:16"), so the new picture
  matches the others.
- **Validated before anything is made:** new words for scene 1 make a new hook (checked for length), and a new last
  scene makes a new call to action.
- **A new hero picture asks about a new paid clip first.** With no answer it exits 3 and does nothing; with "no"
  the scene falls back to Ken Burns and the old clip is archived.
- **It refuses an unfinished run:** "a rerun now would be undone by the graph's next steps". It never touches the
  checkpoint, so a finished run stays finished.

---

## 16. Errors that happened, and what they teach
| # | What happened | Lesson |
|---|---|---|
| E1 | a **Phase 5 test started failing by itself** on a new day: the quota log used the wall clock, the test a fixed clock | a test with a fixed clock must not let the code read the real one anywhere |
| E2 | a check was **silently always False**: Typer 0.27 carries **its own copy of Click** (`typer._click`), so `click.core.ParameterSource` from the real `click` package never equals Typer's | when a library vendors a dependency, "the same" classes aren't the same; compare by name |
| E3 | `--RUN  RUN` in help | a metavar that equals the upper-cased name confuses Typer's Rich help; renamed to `ID` |
| E4 | **a brand-new run "spent $0.0662"** before its first node | **Pydantic's `default_factory` also runs when *parsing*.** `run_id`'s default read the *current* run from a context variable, so every old ledger line (Phases 1–5, no `run_id`) read back as this run's. Fixed by parsing with the variable cleared. A context-dependent default is a hidden input to parsing |
| E5 | a live run showed as "crashed" | a checkpoint can't tell running from dead → the `.running` marker with a process id |
| E6 | **PowerShell 5.1 garbled redirected Uzbek** | §14; fixed with `-o FILE` and ASCII JSON. Test in the user's real terminal |
| E7 | blank lines and broken columns in Markdown tables (a `\|` inside a cell) | a separator character inside data breaks the format |
| E8 | `--plain` in a terminal wrapped every row (fixed width 200) | plain on a terminal keeps the terminal's width |
| E9 | MLflow printed "Flushing the async trace logging queue…" after every command | turn library noise down (`MLFLOW_LOGGING_LEVEL=WARNING`) |
| E10 | `shortsmith mcp` lost the server's startup log line | the CLI's logger setting was inherited by the server's child logger |
| E11 | an error message wasn't shortened (`": "` vs `"; "`) | found by its own test |
| E12 | 5 test bugs; one pushed a real UX fix: a paused run said "no script.json" before "not finished" | check the most useful condition first |

**Usability, from `--help` only.** The build session tried the tool as a new user and changed 12 things. Examples:
- descriptions cut mid-sentence → one-line summaries, and commands grouped;
- noisy `--research / --no-research` switches → single flags;
- `<float range>` → `USD`;
- `--debug` typed at the end failed → it now works anywhere;
- a long free-tier wait looked frozen → the spinner shows elapsed time.

**Measured, not fixed:** `shortsmith list` takes ~5–7 s to start, because importing openai, edge_tts and LangGraph
is slow. The listing itself takes 0.05 s.

**Open items** (for later):
- the writer's retry isn't in the budget estimate;
- **a live y/N answer typed by a person was never tested** (the build's terminal can't type into a running
  program). Worth one try by you:
  ```bash
  uv run shortsmith make "your topic"
  ```
  then answer the hero-clip question;
- `--real` never run;
- more than 1 hero clip;
- YouTube's broader query drops the *last* word ("tea houses of Samarkand" → "tea houses of", losing the place);
- English videos run long (34.4 s), and the critic loop still doesn't converge;
- slow start-up.

---

## 17. Tests, and the optional exercise

- **391 tests** (+68), ~70 s:
  - test_cli 43, test_budget 16, test_rerun 8, plus a ledger regression test for E4;
  - all offline, with Phase 6's fakes and Typer's `CliRunner` (stdout and stderr kept separate);
  - real LangGraph and SQLite in a temp folder.
- **What they prove:**
  - the help tree, exit codes and settings order;
  - `make` → final.mp4 with results on stdout, progress on stderr, a Markdown table and **no ANSI codes**;
  - pause → exit 3 → `resume --approve no` → "already finished";
  - **the budget stop** (exit 4 before the critic), then a resume with no new writer call;
  - `SHORTSMITH_BUDGET` from the environment, overridden by a flag;
  - `--real` refused without `--yes` (no folder, no call);
  - `--mock` beats `MOCK_MEDIA=0`;
  - a missing key fails before any folder;
  - Ctrl+C → 130;
  - every rerun-scene case;
  - `-o FILE` writes strict UTF-8 and the JSON is ASCII;
  - `doctor` never prints a key;
  - `NO_COLOR` and `--plain`;
  - `python -m shortsmith.make --help` still works.
- **Optional exercise** (Phase 7 has no 🧠 step): `exercises/phase-7/budget_rules.py`. It covers the rules behind
  `--budget` and the exit codes, with numbers small enough to check on paper:
  1. `estimate_call(past, first_call)`: the maximum of the last 10 calls;
  2. `would_exceed(spent, estimate, limit)`: the stop rule (a free step never stops);
  3. `suggest_budget(spent, remaining)`: rounded **up** to a whole cent;
  4. `exit_code(outcome)`.

  ```bash
  uv run pytest exercises/phase-7
  ```
  18 tests, all red now.

**What Phase 7 cost:** 14 LLM calls + 1 embedding request, $0.0485 list price, **$0 billed**. gemini-3.8-flash
used 4 of its 20 daily requests, and YouTube 202 units.

---

## Check questions (optional, answer whenever you like)
1. What's the difference between an argument, an option and a flag? Which should `<run>` be, and which `--lang`?
2. What does `[project.scripts] shortsmith = "shortsmith.cli:app"` actually create on disk?
3. Why does Typer need type hints, and what does it do with the docstring?
4. Why should "paused for approval" and "over budget" have different exit codes from "error"?
5. Why must `--json` output go to stdout and progress to stderr? Which Phase 4 rule is this?
6. What are ANSI escape codes, and why must a CLI drop them when its output isn't a terminal?
7. A flag says `--mock`, `.env` says `MOCK_MEDIA=0`. Which wins, and why is that the right order?
8. Why can't `--budget` guarantee a limit to the cent? What does it guarantee instead?
9. What should the CLI do when it needs a yes/no answer and nobody is at the keyboard?
10. How does `rerun-scene` avoid paying for the other five scenes?

**Our build**
11. Why does the budget check live in the graph's node wrapper, not in `cli.py`?
12. Why is the estimate the *maximum* of the last 10 calls, not the average? When was it still wrong?
13. A run stopped as "spent $0.0662" before its first node. What did Pydantic's `default_factory` have to do with it?
14. Why can't a checkpoint alone tell "running" from "crashed"? What fixed it?
15. Why did `> file.txt` garble Uzbek text in your PowerShell but `-o file.txt` didn't?
16. Why is `--json` ASCII-only?
17. What happens with `--real` when there's no terminal and no `--yes`, and why before any folder is made?
18. Why does `doctor` make no network call?

## Interview angles
- "I wrapped the agent system in a Typer CLI: commands for make, resume, list, show, rerun-scene and cost, with
  `--json` for scripts and clear exit codes, so it's usable by people and by other programs."
- "Results go to stdout and progress to stderr, colours and spinners only when there's a terminal, the same
  channel discipline an MCP stdio server needs."
- "Money is guarded at the interface: safe defaults, a per-run budget that stops cleanly and resumes, and
  confirmation before any real paid media."
- "The budget check sits in the graph, before every node, with a conservative estimate from the ledger's recent
  history, so any front end, CLI or web API, gets the same hard stop, and a stopped run resumes without repaying."
- "I tested output where the user actually runs it: Windows PowerShell 5.1 re-encodes redirected output with the
  console code page, so I added `-o FILE` and ASCII-only JSON."
- "A Pydantic `default_factory` that read a context variable also ran during parsing, so old ledger lines were
  counted as the current run's spending. I found it because a fresh run hit its budget before doing anything."
