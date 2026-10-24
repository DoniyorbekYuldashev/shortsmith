# Phase 7 build notes (Steps 7.1-7.3: the `shortsmith` command)

Written by the build session for the teaching session. Everything below happened on this machine (Windows 11
Home with Smart App Control on, Windows PowerShell 5.1.26100 with the console on code page IBM437, Python
3.12, uv 0.13.1, **typer 0.27.3** (which carries its own copy of Click), **rich 15.0.0**, python-dotenv 1.2.4,
langgraph 1.2.16, mlflow 3.17.2) on 2026-10-24. Money spent: **$0** (Gemini free tier for every LLM call,
placeholder images, mock hero clips, edge-tts, local ffmpeg, YouTube within our cap).

**Done-check, in one paragraph.** `uv run shortsmith --help` shows a command tree in three groups (Make videos:
`make`, `resume`, `rerun-scene`; Look at runs: `list`, `show`, `cost`; Setup: `doctor`, `mcp`), the settings
order and the exit codes. `shortsmith make "topic"` runs the Phase 6 graph with a spinner per node (elapsed time,
estimated cost) and one line per finished node (seconds, tokens, list price, what it decided), then a summary
table on stdout. **Live run 1** (tea houses of Samarkand, uz, `--research`): 11 calls, $0.0355 list price,
$0 billed, critic 5.5 -> 6 -> 6, 761 s (the three critic calls waited 135, 165 and 266 s in Gemini's free queue).
**`rerun-scene`** on it remade only scene 2's image and voice (0 calls, $0) and kept the old files in
`history/rerun_1/`. **Live run 2** (Tashkent metro, en) was stopped by `--budget 0.01` before the critic
($0.0071 spent + ~$0.0074 estimated > $0.0100; exit 4), `list` showed "over budget before critic", and
`resume --budget 0.02` finished it with one critic call ($0.0131 total). `list`, `show`, `cost` print tables
(Rich in a terminal, Markdown when piped) and `--json` (ASCII-only JSON) for scripts; `-o FILE` writes UTF-8
itself because PowerShell 5.1 garbles our UTF-8 when it pipes or redirects it (E6). `--real` shows an
estimate and asks, refuses without `--yes` when nobody can be asked (exit 2), and was never run for real.
`python -m shortsmith.make` and `python -m shortsmith.mcp_server` work as before. **391 tests pass** (323 before
+ 68); `exercises/phase-7/` has 18 tests that fail with NotImplementedError by design.

**Three things the lesson should not miss**, all found by running it: (1) **a Pydantic `default_factory` also
runs when a line is parsed**: `LedgerEntry.run_id`'s factory reads the current run from a context variable, so
inside a node every old ledger line (all of Phases 1-5) read back as the current run's, and a brand-new run
was stopped as "spent $0.0662" (E4); (2) **Typer 0.27 vendors Click**: `import click` gets a different Click,
whose classes never match Typer's, so a check like `source == click.core.ParameterSource.COMMANDLINE` is
silently always False (E2); (3) **Windows PowerShell 5.1 re-decodes a native program's output in the console
code page** when it pipes or redirects it: our UTF-8 `—`, `ʻ`, `☕` became `ΓÇö`, `╩╗`, `Γÿò` in a UTF-16 file
(E6). The fixes: parse with the context variable cleared, compare by enum name, and `-o FILE` / ASCII JSON.

**Changes for the teaching session's files** (I did not edit them):
- CLAUDE.md "Current progress" and the commands: `uv run shortsmith make "topic"` is now the way to make a video
  (`python -m shortsmith.make` still works); `shortsmith doctor` checks the setup without printing keys;
  exit codes 0/1/2/3/4/130; `.env` is loaded by the command itself (so `uv run shortsmith ...` works without
  `--env-file`; `uv run --env-file .env shortsmith ...` also works). Gotcha worth adding: in PowerShell 5.1 use
  `-o FILE` (or `--json`) instead of `>` for non-English output.
- PLAN Phase 7: the done-check holds; the flags grew (`--budget` counts list price; `--hero-clips` is 0 or 1;
  `--mock/--real` with a confirmation). PLAN §4 (repo tree): new `cli.py`, `console.py`, `views.py`, `runs.py`,
  `costs.py`, `budget.py`, `rerun.py`, `__main__.py`. PLAN §5: `RunState` has three more fields
  (`budget_usd`, `hero_clips`, `media`).
- `docs/lessons/phase-7.md` exists (untracked, the teaching session's): not touched, not committed.

---

## 1. What was built

### Files

| File | Status | What it is |
|---|---|---|
| `pyproject.toml`, `uv.lock` | changed | `uv add typer rich python-dotenv` (all three were already in the lock through `mcp[cli]`: no download, now direct dependencies). `[project.scripts] shortsmith = "shortsmith.cli:main"` (was `shortsmith:main`, uv init's hello stub) |
| `src/shortsmith/cli.py` | new | the Typer app: the 8 commands, flags, `ExitCode`, error -> message + hint (`explain`, `failures`), `preflight`, the `--real` question, `-o FILE`, `main()` (UTF-8 streams, `.env`, logging, global flags anywhere) |
| `src/shortsmith/console.py` | new | `Ui` (stdout and stderr consoles, rich or plain per stream, Markdown tables when piped), `NodeProgress` (spinner + done lines), the approval and yes/no questions |
| `src/shortsmith/views.py` | new | how results look: the make summary, `show`, `cost` (data in, tables out; split from cli.py so the commands stay readable) |
| `src/shortsmith/runs.py` | new | the runs as data: `RunSummary`, `RunDetail`, status from the checkpoint, `resolve("latest" / prefix / path)`, the `.running` marker, trace ids from mlflow.db |
| `src/shortsmith/costs.py` | new | the ledger by day / run / model and step, list vs billed, the quota view (Gemini since midnight Pacific, YouTube units) |
| `src/shortsmith/budget.py` | new | estimates per node, `BudgetExceeded`, `check()`, `run_estimate()`, `nodes_left()` |
| `src/shortsmith/rerun.py` | new | `plan_change()` and `rerun_scene()`: one scene through the producer's manifest, `reruns.json` |
| `src/shortsmith/__main__.py` | new | `python -m shortsmith` = the command |
| `src/shortsmith/__init__.py` | changed | the hello-world `main()` is gone (a docstring now) |
| `src/shortsmith/agents/graph.py` | changed | node wrapper: `budget.check()` before the node, then the `ON_START` hook (context variable); `drive(on_start=)`; `new_state(budget_usd, hero_clips, media)`; `new_run_id()` (a second run in the same second gets `-2`); `describe()` split out of `_progress` (legacy output byte for byte the same); `--hero-clips 0` in the director node |
| `src/shortsmith/agents/state.py` | changed | `budget_usd`, `hero_clips`, `media` (saved with the run; a resume keeps them unless its flags change them) |
| `src/shortsmith/agents/producer.py` | changed | `pending()` (what a pass would make, for the estimate), `produce(only=...)`, `archive()` with a label (`rerun_N`), `adopt()` for runs from before media.json, `image_stale`/`voice_stale` shared by both; the MCP SDK imported only with `--mcp` (0.5 s) |
| `src/shortsmith/agents/director.py` | changed | `ken_burns_only()` |
| `src/shortsmith/ledger.py` | changed | `entries()` parses with `CURRENT_RUN` cleared (E4) |
| `src/shortsmith/research/youtube.py` | changed | quota lines logged at the search's own `now` (E1) |
| `src/shortsmith/tracing.py` | changed | `MLFLOW_LOGGING_LEVEL=WARNING` (E9) |
| `src/shortsmith/mcp_server.py` | changed | `main(argv=None)`, so `shortsmith mcp` can call it |
| `src/shortsmith/make.py` | changed | `new_run(run_dir=...)` (so `--simple` can tag the ledger), a docstring pointer to the new command; otherwise as it was |
| `README.md` | changed | a Usage section |
| `tests/test_cli.py` (43), `test_budget.py` (16), `test_rerun.py` (8) | new | §6 |
| `tests/test_ledger.py` (+1) | changed | E4's regression test |
| `exercises/phase-7/budget_rules.py` + `test_budget_rules.py` | new | optional: `estimate_call`, `would_exceed`, `suggest_budget`, `exit_code`; 18 tests |

### The command tree

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
│  ├─ show [RUN]        [--json] [-o FILE]                      (RUN defaults to latest)
│  └─ cost              [--days N] [--run ID] [--top N] [--json] [-o FILE]
└─ Setup
   ├─ doctor
   └─ mcp               [--root FOLDER]
```
`RUN` is a run id, a unique start of one (`2026-10-23`), `latest`, or a folder path. Hidden: `make --crash-before
NODE` (the Phase 6 resume demo). `--debug` and `--plain` also work after the command (`main()` moves them to the
front: the error hint says "run again with --debug", and people add it at the end).

### Every flag of `make`, and where it comes from

| flag | default | env var | notes |
|---|---|---|---|
| `TOPIC` | (required) | | blank -> usage error |
| `--lang/-l` | en | `SHORTSMITH_LANG` | |
| `--research` | off | | +1 LLM call, +1-2 embedding requests, YouTube (cached 24 h) |
| `--brand NAME` | none | | implies `--research` |
| `--approve yes\|no` | ask (terminal) or pause | | |
| `--hero-clips N` | 1 | | 0 or 1; 2+ is a usage error that says why |
| `--budget USD` | none | `SHORTSMITH_BUDGET` | list price, whole run, saved with it |
| `--mock/--real` | mock | `MOCK_MEDIA` (1 = mock, 0 = real) | `--real` asks after an estimate |
| `--yes/-y` | off | | confirms `--real` |
| `--max-rounds N` | 2 | | 0-5 |
| `--pass-score SCORE` | 7.0 | | 0-10 |
| `--critic-model MODEL` | gemini-3.8-flash | `SHORTSMITH_CRITIC_MODEL` | the 20-a-day fallback is gemini-3.6-flash |
| `--prompt v1\|v2` | v2 | | |
| `--mcp` | off | | the producer's tools over the MCP server |
| `--no-trace` | off | | MLflow off |
| `--simple` | off | | the Phase 2 straight pipeline; graph-only flags given with it are a usage error |
| `--music FILE` | generated bed | | `--simple` only |

`resume` takes `--approve`, `--budget` (saved into the checkpoint), `--mock/--real` (default: what the run
started with, NOT the environment), `--yes`, `--critic-model`, `--no-trace`. `rerun-scene`'s `--mock/--real`
also defaults to the run's own media.

### Settings order (in `--help`, the README and cli.py's docstring)

**flags > environment variables > `.env` in the current folder > defaults.** `main()` calls
`load_dotenv(".env", override=False)` before Typer parses anything, so a real environment variable wins over
`.env`, and Typer's `envvar=` reads the result; a flag wins over both. The help shows `[env var: ...]` and
`[default: ...]` on each option. `.env` is read into the process, never printed (doctor says only "found").

### Exit codes (`cli.ExitCode`)

| code | meaning | e.g. |
|---|---|---|
| 0 | ok | a finished run; `resume` of a run that already finished ("nothing to do") |
| 1 | error, with one message and a `next:` hint | unknown run, missing key, a node failed, `--real` declined |
| 2 | usage | bad flag value, `--hero-clips 2`, `--real` without `--yes` and no terminal, `--simple --max-rounds 1` |
| 3 | paused, waiting for an approval | no `--approve` and no terminal; rerun of the hero picture without an answer |
| 4 | stopped: over budget | the next priced step would pass `--budget` |
| 130 | interrupted (Ctrl+C) | 128 + SIGINT; the hint names the resume command |

### How `--budget` works (budget.py), and where it can be wrong

- **Before every node**, the node wrapper calls `budget.check(name, state)`: it reads the ledger, estimates the
  node, and raises `BudgetExceeded` if `spent + estimate > budget_usd`. The exception leaves the node before
  anything is sent; LangGraph keeps the checkpoint of the last finished node and stores the exception in it
  (as its repr string: `"BudgetExceeded('over budget before critic: ...')"`, which is how `list` knows the
  status). The CLI catches it, prints the numbers, suggests a budget that would finish, and exits 4.
- **spent** = the run's own ledger lines (run_id-tagged), across every process that worked on it.
- **estimate**: for each call a node makes, the most expensive of the last 10 calls of the same ledger step on
  the same model (writer = `writer` + `writer-pick`; critic on the critic model). No history -> a first-call
  guess from Phase 6's token counts at today's list price. The producer: `producer.pending()` priced at
  `MEDIA_PRICES` ($0 with mock media). approve_hero and editor are free and never stopped.
- **The budget counts list price** and the output says so every time ("the Gemini free tier bills $0; the
  budget counts what it would cost on a paid tier").
- **Persisted**: `budget_usd` is a RunState field; `resume --budget X` writes it with `app.update_state()` (a new
  checkpoint version; `next` is unchanged, checked for a stop before writer/director/critic and for a paused
  interrupt in a prototype first, §5 D6).
- **Where it can be wrong**: a writer whose 3 options all break a rule writes again (live run 1: 3 calls,
  $0.0092 vs the 2-call estimate $0.0063); a critic that thinks longer than usual; a model with no history (the
  guess). The check runs again before the next node, so a run can pass its budget by at most one node's error.
  Measured, estimate vs actual (list price): researcher $0.0014 / $0.0012, writer $0.0063 / $0.0092 (retry),
  $0.0063 / $0.0055, director $0.0016 / $0.0015 and $0.0016, critic (3 calls) $0.0196 / $0.0205, critic $0.0074 /
  $0.0059, revise (2) $0.0029 / $0.0031.
- **The suggested budget** after a stop = spent + every LLM node still allowed (all remaining critic rounds and
  revisions), rounded up to a cent: a budget that surely finishes, not the cheapest one.

### Other behaviour worth knowing

- **Running vs crashed** (E5): the CLI writes `runs/<id>/.running` with its process id while it drives a run and
  removes it in a `finally`. `list` says "running (critic)" when that process is alive, "crashed at critic" when
  it is gone (a hard kill leaves the marker behind). `resume` refuses a run another live process is driving.
  Windows has no `os.kill(pid, 0)`, so `pid_alive()` asks the kernel for the exit code (`OpenProcess` +
  `GetExitCodeProcess` == STILL_ACTIVE) through ctypes.
- **rerun-scene** never touches the checkpoint (a finished run stays finished); it refuses an unfinished run
  ("a rerun now would be undone by the graph's next steps"); a new prompt keeps the run's style words
  (`reviser.style_of` + `director.with_style`); new words for scene 1 make a new hook, for the last scene a new
  call to action, both validated before anything is made; a changed hero picture asks about a new clip first
  (no answer -> exit 3, nothing done; "no" -> the director's Ken Burns move, the old clip archived). Runs from
  before the manifest (the straight pipeline) are "adopted" first: their files count as made from their script.json.
- **Status** comes from the checkpoint only (plus the marker): finished / running / paused at approval / over
  budget before X / crashed at X / no checkpoint.
- **Trace ids** come from mlflow.db by plain SQL (`trace_info` joined with `trace_tags` where key = 'run_id'):
  importing mlflow takes seconds. If MLflow changes its tables, only the trace ids go missing (a warning).

---

## 2. Real terminal output

### `shortsmith --help` in a terminal (Rich)

```
 Usage: shortsmith [OPTIONS] COMMAND [ARGS]...

 Shortsmith: AI agents that turn one line of text into a ~30-second vertical short video.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --debug              Show the full traceback when something fails. [env var: SHORTSMITH_DEBUG]                       │
│ --plain              Plain output: no colours, spinners or box lines (automatic when the output is piped or          │
│                      redirected).                                                                                    │
│ --version            Print the version and exit.                                                                     │
│ --help     -h        Show this message and exit.                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Make videos ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ make         Make a video from a topic (writer, director, critic loop).                                              │
│ resume       Continue a stopped or paused run from its checkpoint.                                                   │
│ rerun-scene  Redo one scene (image and/or voice) and re-cut the video.                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Look at runs ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ list         The runs, newest first, with status and cost.                                                           │
│ show         One run in detail: script, critic, cost, files.                                                         │
│ cost         Spending by day, run and model; free-tier usage today.                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Setup ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ doctor       Check ffmpeg, keys (set or not), media mode and stores.                                                 │
│ mcp          Start the MCP server on stdio (for Claude Desktop).                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 Settings, highest first: flags > environment variables > .env in the current folder > defaults.

 Exit codes: 0 ok, 1 error, 2 usage, 3 paused for an approval, 4 over budget, 130 interrupted.

 Money: every call is priced at list price in runs/ledger.jsonl; on the Gemini free tier it is billed $0. Images stay
 free placeholders unless you pass --real.
```

### `shortsmith make --help`, piped (plain Click help: `main()` switches Typer's Rich help off when stdout isn't a terminal)

```
Usage: shortsmith make [OPTIONS] {TOPIC}

  Make a video from a topic: research (optional), writer, director, hero-clip
  approval, producer, editor and a critic loop, saved after every step.

  Example: shortsmith make "coffee shop in Tashkent, cozy autumn vibe" --lang
  uz --research

Arguments:
  TOPIC  What the video is about, in a few words.  [required]

Options:
  -l, --lang <en|uz|ru>       Language of the video.  [env var:
                              SHORTSMITH_LANG; default: en]
  --research                  Ground the script in this month's YouTube trends
                              and the hook library (1 more LLM call).
  --brand NAME                Write for a brand in data/hooks/brands/, e.g.
                              anor-va-dolchin (implies --research).
  --approve <yes|no>          Answer the hero-clip question in advance.
                              Without it you are asked; with no terminal the
                              run pauses.
  --hero-clips N              AI hero clips: 1 = the director's one approval-
                              gated clip, 0 = Ken Burns only.  [default: 1]
  --budget USD                Hard limit for the whole run in US dollars, LIST
                              price (e.g. 0.05). Checked before every priced
                              step; the run stops cleanly and can be resumed.
                              [env var: SHORTSMITH_BUDGET; x>=0.0]
  --mock / --real             --mock: free placeholder images and clips (the
                              default). --real: paid images on fal.ai, after a
                              cost estimate and a yes.  [env var: MOCK_MEDIA]
  -y, --yes                   Confirm --real without asking (for scripts).
  --max-rounds N              Critic -> revise rounds at most.  [default: 2;
                              0<=x<=5]
  --pass-score SCORE          Critic score (0-10) at which the video is done.
                              [default: 7.0; 0.0<=x<=10.0]
  --critic-model MODEL        The critic's model; default gemini-3.8-flash.
                              Each free Flash model allows 20 requests a day,
                              so gemini-3.6-flash is the fallback.  [env var:
                              SHORTSMITH_CRITIC_MODEL]
  --prompt <v1|v2>            Writer prompt version.  [default: v2]
  --mcp                       The producer calls the media tools through the
                              MCP server.
  --no-trace                  Don't record the run in MLflow
                              (sqlite:///mlflow.db).
  --simple                    The straight Phase 2 pipeline instead: one
                              script, images, voice, video. No critic, no
                              checkpoints.
  --music FILE                Your own music file (with --simple); default: a
                              generated bed.
  -h, --help                  Show this message and exit.
```
(Typer's plain help writes a required argument as `{TOPIC}` in the usage line; Click itself would write `TOPIC`.)

### A full `make` run in a terminal (live run 1, `2026-10-24_182435`, Rich mode)

`uv run shortsmith make "tea houses of Samarkand, slow mornings" --lang uz --research --approve yes`. While a node
runs, one spinner line redraws in place, e.g. `⠦ critic: watching 4 frames and judging the video (est. $0.0051)
0:01:09`; when it finishes, the spinner is replaced by its `done` line:

```
run 2026-10-24_182435 | 'tea houses of Samarkand, slow mornings' (uz) | prompt v2 | up to 2 revisions, pass at 7 | hero
clips 1 | research
media: mock placeholders (free) | LLM: gemini-3.5-flash-lite, critic gemini-3.8-flash
done researcher       8.7 s   1,522 in    298 out   $0.0012  notes: 2 hook patterns, 2 angles
done writer          13.1 s   2,355 in  3,402 out   $0.0092  3 options, picked 1: "Tonggi Samarqandni ko'ring." |
rejected: 3
done director         5.0 s     750 in    505 out   $0.0015  hero = scene 4: Steam rises gracefully from the spout of
the ceramic teapot as hot water is poured, while the camera slowly pushes in closer.

hero clip? Scene 4 ("Qaynoq suv quyilgach, xushbo'y hid butun ayvonga tarqaladi.") as a 5-second AI clip (Steam rises
gracefully from the spout of the ceramic teapot as hot water is poured, while the camera slowly pushes in closer.) costs
~$0.30 with real media; MOCK_MEDIA=1 makes a free placeholder clip now. Continue?
  answered by --approve: yes
done approve_hero     0.0 s                                hero clip approved
done producer        14.7 s                                round 0: images 1, 2, 3, 4, 5, 6; voices 1, 2, 3, 4, 5, 6;
clip 4
done editor          28.8 s                                runs\2026-10-24_182435\final.mp4
done critic         135.4 s   5,379 in    828 out   $0.0071  round 0: score 5.5, weak [1] (frames of scenes [1, 3, 4,
6])
done revise           1.7 s     804 in    478 out   $0.0014  scene 1 narration
done approve_hero     0.0 s                                no new clip needed
done producer         5.2 s                                round 1: voices 1
done editor          57.6 s                                runs\2026-10-24_182435\final.mp4
done critic         165.4 s   5,393 in    895 out   $0.0074  round 1: score 6, weak [5] (frames of scenes [1, 3, 4, 6])
done revise           6.2 s     822 in    564 out   $0.0017  scene 5 narration
done approve_hero     0.0 s                                no new clip needed
done producer         2.2 s                                round 2: voices 5
done editor          48.7 s                                runs\2026-10-24_182435\final.mp4
done critic         266.3 s   5,400 in    504 out   $0.0059  round 2: score 6, weak [6] (frames of scenes [1, 3, 4, 5])
run 2026-10-24_182435
 node           calls   tokens in   tokens out   list price   estimate   seconds
─────────────────────────────────────────────────────────────────────────────────
 researcher         2       1,538          298      $0.0012    $0.0014       8.7
 writer             3       2,355        3,402      $0.0092    $0.0063      13.1
 director           1         750          505      $0.0015    $0.0016       5.0
 approve_hero                                                                0.0
 producer                                                                   22.1
 editor                                                                    135.1
 critic             3      16,172        2,227      $0.0205    $0.0196     567.2
 revise             2       1,626        1,042      $0.0031    $0.0029       7.8
cost     $0.0355 list price | $0 billed (Gemini free tier) | 11 calls (11 in this session) | 29,915 tokens
trace    tr-2c33537d5518d462f8c45f44917627a1 (MLflow: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port
5000)
video    D:\All codes\shortsmith\runs\2026-10-24_182435\final.mp4  (27.1 s, 1080x1920, 3.9 MB)
critic   5.5 -> 6 -> 6 | max rounds (2 revisions; last score 6)
done     in 761 s. Details: shortsmith show 2026-10-24_182435 | redo a scene: shortsmith rerun-scene 2026-10-24_182435
<scene> --prompt "..."
```
Reading it: progress (the header, `done` lines, the question) went to stderr; the table and the lines after it
to stdout. "rejected: 3" = the writer's first attempt had all three Uzbek options outside the word rule, so it
wrote again (the estimate missed exactly that). `researcher 2 calls` = the chat call + one embedding request.
The critic estimate grew from $0.0051 to $0.0071 to $0.0074 during the run: each call became history for the next.
The loop didn't converge (5.5 -> 6 -> 6: a new complaint each round, as in Phase 6).

### `rerun-scene` (on run 1, live; non-ASCII text on purpose for the UTF-8 checks)

```
> uv run shortsmith rerun-scene 2026-10-24_182435 2 --prompt "Medium shot of a quiet courtyard teahouse at dawn, a teapot ☕ steaming on a low wooden table" --narration "Qadimgi hovlida tong salqini — oʻzgacha sokinlik."
rerun scene 2 of 2026-10-24_182435: image and voice | media: mock placeholders (free)
scene 2: making the image and voice, then cutting the video...
made     images 2; voices 2 (scene 2; every other scene untouched)
picture  Medium shot of a quiet courtyard teahouse at dawn, a teapot ☕ steaming on a low wooden table, cinematic, photorealistic, 9:16
words    Qadimgi hovlida tong salqini — oʻzgacha sokinlik.
kept     the replaced files and the old final.mp4 in runs\2026-10-24_182435\history\rerun_1
video    D:\All codes\shortsmith\runs\2026-10-24_182435\final.mp4
cost     $0 list price
```
Checked afterwards: only `scenes/scene_02.png` and `audio/scene_02.mp3` had new times (18:38; scene 1's and 5's
voices kept their critic-round times 18:28 and 18:32), `history/rerun_1/` held `scenes/scene_02.png`,
`audio/scene_02.mp3` and `final.mp4`, and the ledger still had 184 lines (no call). ", cinematic, photorealistic,
9:16" was added: the run's style words, so the new picture matches the others.

### The budget stop, piped (live run 2, plain mode: `cmd /c "uv run shortsmith ... > file 2>&1"`)

The first attempt stopped before the writer because of E4 (below). After the fix, `resume --budget 0.01`:
```
resuming 2026-10-24_184531 before writer | 'why Tashkent metro stations look like palaces' (en) | prompt v2 | up to 0 revisions, pass at 7 | hero clips 1
media: mock placeholders (free) | LLM: gemini-3.5-flash-lite, critic gemini-3.8-flash
budget: $0.0100 for the whole run, in LIST price (the Gemini free tier bills $0; the budget counts what it would cost on a paid tier)
starting MLflow tracing...
  ... writer: writing 3 script options, then picking one (est. $0.0063)
  done writer          14.2 s   1,162 in  2,064 out   $0.0055  3 options, picked 1: 'Ever rode a subterranean palace?'
  ... director: planning the shots and the hero scene (est. $0.0016)
  done director         6.0 s     700 in    566 out   $0.0016  hero = scene 1: Commuters walk past the ornate marble pillars while the camera slowly pushes forward into the grand underground station.
  ... approve_hero: checking whether a hero clip needs a yes

hero clip? Scene 1 ("Ever rode a subterranean palace? Tashkent hides architectural masterpi") as a 5-second AI clip (Commuters walk past the ornate marble pillars while the camera slowly pushes forward into the grand underground station.) costs ~$0.30 with real media; MOCK_MEDIA=1 makes a free placeholder clip now. Continue?
  answered by --approve: yes
  ... approve_hero: checking whether a hero clip needs a yes
  done approve_hero     0.0 s                                hero clip approved
  ... producer: making the images and voices that are missing
  done producer        54.8 s                                round 0: images 1, 2, 3, 4, 5, 6; voices 1, 2, 3, 4, 5, 6; clip 1
  ... editor: cutting the video with ffmpeg
  done editor          55.7 s                                runs\2026-10-24_184531\final.mp4
run 2026-10-24_184531
| node         | calls | tokens in | tokens out | list price | estimate | seconds |
|--------------|-------|-----------|------------|------------|----------|---------|
| writer       |     2 |     1,162 |      2,064 |    $0.0055 |  $0.0063 |    14.2 |
| director     |     1 |       700 |        566 |    $0.0016 |  $0.0016 |     6.0 |
| approve_hero |       |           |            |            |          |     0.0 |
| producer     |       |           |            |            |          |    54.8 |
| editor       |       |           |            |            |          |    55.7 |
cost     $0.0071 list price | $0 billed (Gemini free tier) | 3 calls (3 in this session) | 4,492 tokens
budget   $0.0071 of $0.0100 used (list price)
trace    tr-aa0629604ce18fd002ac473ab0ab2614 (MLflow: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000)
stopped  over budget before critic after 132 s
         spent $0.0071 + critic ~$0.0074 > budget $0.0100 (list price; the free tier bills $0). State saved; nothing was sent for that step.
next     shortsmith resume 2026-10-24_184531 --budget 0.02
```
Exit code 4. That is also the piped/plain-mode sample: no colours, no spinner, a `...` line when a node starts (so a
log file shows what was running when something hung), `done` lines, a Markdown table. `approve_hero` starts twice:
an interrupt re-runs its node from the top after the answer (Phase 6, D12). `list` right after:
`| 2026-10-24_184531 | Palaces Underground | en | - | 34.4 s | $0.0071 | over budget before critic |`.

### ... and the resume that finished it (terminal, Rich)

```
resuming 2026-10-24_184531 before critic | 'why Tashkent metro stations look like palaces' (en) | prompt v2 | up to 0
revisions, pass at 7 | hero clips 1
media: mock placeholders (free) | LLM: gemini-3.5-flash-lite, critic gemini-3.8-flash
budget: $0.0200 for the whole run, in LIST price (the Gemini free tier bills $0; the budget counts what it would cost on
a paid tier)
done critic         132.2 s   5,389 in    500 out   $0.0059  round 0: score 6.2, weak [1, 6] (frames of scenes [1, 3, 4,
6])
run 2026-10-24_184531
 node           calls   tokens in   tokens out   list price   estimate   seconds
─────────────────────────────────────────────────────────────────────────────────
 writer             2       1,162        2,064      $0.0055                 14.2
 director           1         700          566      $0.0016                  6.0
 approve_hero                                                                0.0
 producer                                                                   54.8
 editor                                                                     55.7
 critic             1       5,389          500      $0.0059    $0.0074     132.2
cost     $0.0131 list price | $0 billed (Gemini free tier) | 4 calls (1 in this session) | 10,381 tokens
budget   $0.0131 of $0.0200 used (list price)
video    D:\All codes\shortsmith\runs\2026-10-24_184531\final.mp4  (34.4 s, 1080x1920, 5.5 MB)
critic   6.2 | max rounds (0 revisions; last score 6.2)
```
One new call: the writer and director were not paid for again. The "estimate" column is filled only for nodes run
in this session.

### `list` (`-o` file, the same as piped stdout)

```
| id                       | title                          | lang | score | length |    cost | status        |
|--------------------------|--------------------------------|------|-------|--------|---------|---------------|
| 2026-10-24_184531        | Palaces Underground            | en   |   6.2 | 34.4 s | $0.0131 | finished      |
| 2026-10-24_182435        | Samarqand tongi                | uz   |     6 | 26.7 s | $0.0355 | finished      |
| 2026-10-24_001351        | Campus ID Goldmine             | en   |     8 | 29.2 s | $0.0116 | finished      |
| 2026-10-23_235059        | Registon kechasi               | uz   |   7.8 | 25.2 s | $0.0178 | finished      |
| 2026-10-23_234601-mcp    | Cozy Autumn Coffee in Tashkent | en   |     - |      - |       - | no checkpoint |
| 2026-10-23_233023-vision | Cozy Autumn Coffee in Tashkent | en   |     - | 33.3 s |       - | no checkpoint |
```
On stderr after it: `6 of 19 runs; 13 older: --limit 0. Cost = list price of the run's ledger lines (billed $0 on
the Gemini free tier; '-' = made before runs were tagged). Details: shortsmith show <id>`.

### `show` (the head; plain)

```
Palaces Underground (en)
run      2026-10-24_184531 | topic 'why Tashkent metro stations look like palaces'
status   finished
settings prompt v2 | research off | up to 0 revisions, pass at 7 | hero clips 1 | media mock | budget $0.0200
hook     Ever rode a subterranean palace?
cta      Follow my page for more hidden travel gems now
hero     scene 1: Commuters walk past the ornate marble pillars while the camera slowly pushes forward into the grand underground station. (approved; ~$0.30 with real media)
Script
| # | motion    | words | voice | narration                                                                         | picture prompt                                                                   |
|---|-----------|-------|-------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| 1 | ai_clip   |    11 | 6.2 s | Ever rode a subterranean palace? Tashkent hides architectural masterpieces deep   | Wide shot of a grand underground Tashkent metro station with ornate white marble |
|   |           |       |       | underground.                                                                      | pillars and glowing crystal chandeliers, commuters walking on the platform,      |
...
```
then Critic (round, score, weak scenes with reason -> fix, frames, model), Revisions (before -> after), Reruns,
Cost per node (calls, tokens, list price, billed, seconds, total), and `video`, `files`, `history`, `traces`
(`tr-2c33537d5518d462f8c45f44917627a1 (OK, 759.9 s)` for run 1) with the command that opens the MLflow UI.

### `cost --days 1 --top 3` (plain)

```
Spending, since 2026-10-24 (runs/ledger.jsonl)
By day (this computer's time zone)
| day        | calls | tokens in | tokens out | list price | billed |
|------------|-------|-----------|------------|------------|--------|
| 2026-10-24 |    54 |    72,001 |     29,150 |    $0.1079 |     $0 |
By run
| run                              | calls | tokens in | tokens out | list price | billed |
|----------------------------------|-------|-----------|------------|------------|--------|
| phase6_prompts_2026-10-24_000657 |    40 |    29,467 |     22,332 |    $0.0704 |     $0 |
| 2026-10-24_001351                |     6 |     9,077 |      2,512 |    $0.0116 |     $0 |
| 2026-10-23_235059                |     1 |     5,432 |      1,294 |    $0.0089 |     $0 |
| ... 2 more (--top)               |       |           |            |            |        |
By model and step
| model                 | step       | calls | tokens in | tokens out | list price | billed |
|-----------------------|------------|-------|-----------|------------|------------|--------|
| gemini-3.5-flash-lite | writer     |    11 |     8,299 |     18,714 |    $0.0493 |     $0 |
| gemini-3.6-flash      | critic     |    12 |    18,685 |      3,169 |    $0.0259 |     $0 |
| gemini-3.5-flash-lite | researcher |     6 |     8,520 |      2,016 |    $0.0076 |     $0 |
| ... 10 more (--top)   |            |       |            |            |            |        |
total    54 calls | 101,151 tokens | $0.1079 list price | $0 billed
billed   Gemini calls are $0 on the free tier (GEMINI_TIER=free, the default); OpenAI and fal.ai lines are billed at list price.
Gemini today (since midnight Pacific = 07:00 UTC)
| model            | requests | free daily limit | left (by our count) |
|------------------|----------|------------------|---------------------|
| gemini-3.6-flash |        0 |               20 |                  20 |
| gemini-3.8-flash |        0 |               20 |                  20 |
YouTube  0 units today (our cap 2,000; Google's free quota 10,000; a live search is 101 units, a cached one 0)
note     Google can't be asked how much free quota is left: these are the calls OUR logs recorded since midnight Pacific (2026-10-24 07:00 UTC).
note     Refused requests (429) are not in the ledger but may still count against the day.
note     flash-lite's daily limit is far higher than Flash's 20 and not measured; see aistudio.google.com/rate-limit.
```
(Taken before the live runs. "By day" is local time: the Phase 6 build's last calls at 19:18 UTC on 10-23 were
00:18 on 10-24 in Tashkent. After the live runs the quota table said gemini-3.8-flash: 4 requests, 16 left.)

### `list --json` (the start; ASCII-only, so `| ConvertFrom-Json` works in PowerShell 5.1)

```json
[
  {
    "id": "2026-10-24_184531",
    "folder": "runs\\2026-10-24_184531",
    "created": "2026-10-24T18:45:31",
    "topic": "why Tashkent metro stations look like palaces",
    "title": "Palaces Underground",
    "language": "en",
    "score": 6.2,
    "reviews": 1,
    "length_s": 34.4,
    "final_mp4": "D:\\All codes\\shortsmith\\runs\\2026-10-24_184531\\final.mp4",
    "calls": 4,
    "tokens": 10381,
    "cost_usd": 0.01305035,
    "billed_usd": 0.0,
    "status": {
      "kind": "finished",
      "node": null,
      "detail": "",
      "label": "finished"
    }
  },
  ...
```
`show --json` is a `RunDetail` (the summary + settings, scenes, hook, cta, hero, critic_rounds, revisions, made,
reruns, nodes, files with ffprobe's `video`, traces); `cost --json` a `CostReport` (total, by_day, by_run,
by_model_step with separate model/step, quotas). Uzbek text comes out as `\u02bb`-style escapes; any JSON reader
turns them back.

### `doctor` (live; key values never printed)

```
| check           |    | detail                                                                             |
| ffmpeg          | ok | ffmpeg version 9.0.2-full_build-www.gyan.dev; C:\Users\doniy\...\bin\ffmpeg.EXE    |
| ffprobe         | ok | ffprobe version 9.0.2-full_build-www.gyan.dev; C:\Users\doniy\...\bin\ffprobe.EXE  |
| .env            | -  | found in this folder (loaded; not printed)                                         |
| LLM provider    | ok | gemini: gemini-3.5-flash-lite; critic gemini-3.8-flash                             |
| GEMINI_API_KEY  | ok | set; LLM calls and embeddings (LLM_PROVIDER=gemini)                                |
| OPENAI_API_KEY  | -  | set; LLM calls (LLM_PROVIDER=openai)                                               |
| FAL_KEY         | -  | set; real images and hero clips (--real)                                           |
| YOUTUBE_API_KEY | -  | set; trend research (--research)                                                   |
| critic price    | ok | gemini-3.8-flash has a list price in ledger.py                                     |
| MOCK_MEDIA      | -  | mock: free placeholders                                                            |
| billing         | -  | GEMINI_TIER=free (what `cost` counts as billed)                                    |
| ledger          | -  | runs\ledger.jsonl: 173 lines, $0.2148 list price                                   |
| checkpoints     | -  | checkpoints\graph.sqlite: 3 runs                                                   |
| runs            | -  | runs/: 17 runs                                                                     |
| MLflow          | ok | installed; mlflow.db: 2.1 MB                                                       |
ready: `shortsmith make "your topic"` costs $0 with mock media on the Gemini free tier
```
(Paths shortened here. Only the current provider's key can FAIL; the others are information. No network call: a
"key works" check would spend a request of the day's quota.)

### Errors, as a person sees them

```
> uv run shortsmith show no-such-run
error: no run 'no-such-run' in runs/: see `shortsmith list`
next: `shortsmith list` shows the runs                                  (exit 1)

> uv run shortsmith make "x" --hero-clips 2
Usage: shortsmith make [OPTIONS] {TOPIC}
Try 'shortsmith make -h' for help.

Error: Invalid value for '--hero-clips': 0 or 1 for now. The video target is 6 images + 1 approval-gated hero clip;
more clips would need the director to choose several scenes, and an approval and a price check for each clip.   (exit 2)
```
From the tests (CliRunner, fakes): `error: RuntimeError: the director tripped` / `next: the run is saved after its
last finished node: shortsmith resume <id>; run again with --debug to see the traceback` (exit 1, `list` says
"crashed at director"); `error: GEMINI_API_KEY is not set (LLM_PROVIDER=gemini)` before any folder exists; `--real`
without `--yes` and no terminal: the estimate lines, then `error: --real needs a yes, and there is no terminal to
ask` / `next: add --yes to confirm the spending` (exit 2, no run folder, no call).

### MCP

`cmd /c "uv run shortsmith mcp < NUL > out.txt 2> err.txt"`: exit 0 when stdin closes, **0 bytes on stdout**, and
the server's own log line on stderr (`INFO shortsmith.mcp: shortsmith MCP server | root D:\All codes\shortsmith |
images: placeholders (MOCK_MEDIA) | LLM provider: gemini (key set) | ffmpeg: ...`) after E10's fix.

---

## 3. Windows: UTF-8, emoji, PowerShell and files (checked live in the owner's terminal)

The owner's terminal: Windows PowerShell 5.1.26100.9711, `[Console]::OutputEncoding` = IBM437. Scene 2's row of run
1 holds `—`, `ʻ` and `☕`.

| how | file starts with | what came out |
|---|---|---|
| straight to the console (no pipe) | | `— oʻzgacha`, `☕`: right (Python writes to the console with the UTF-16 console API) |
| `shortsmith show ... > show.txt` in PowerShell 5.1 | `FF FE` (UTF-16LE) | `ΓÇö o╩╗zgacha`, `Γÿò`: garbled |
| `shortsmith show ... \| Select-String ...` | | garbled the same way |
| `shortsmith show ... -o show.md` | `53 61 6D` (UTF-8, no BOM) | right |
| `cmd /c "shortsmith show ... > show.txt"` | `53 61 6D` (UTF-8) | right |
| `shortsmith show --json \| ConvertFrom-Json` | | right (the JSON is ASCII: `\u02bb`) |
| this build tool's own PowerShell (UTF-8 defaults) `>` | `EF BB BF` | right: a different machine setup gives a different answer |

Why: PowerShell 5.1 doesn't pass a native program's bytes through a pipe or `>`; it decodes them as text with
`[Console]::OutputEncoding` (here IBM437) and re-encodes them (UTF-16 for `>`). Our side was already right: `main()`
reconfigures stdout and stderr to UTF-8, so a redirected stream isn't cp1252 (which would raise
UnicodeEncodeError on Uzbek text); the decode happens after the bytes leave Python. Fixes we can make: `-o FILE`
(Python writes the file itself) and ASCII-only JSON. Fixes on the user's side: `[Console]::OutputEncoding =
[Text.UTF8Encoding]::new()` before redirecting, or `cmd /c`. The README says so.

---

## 4. Errors hit, how they were found, how they were fixed

**E1. `assert youtube.units_today(paths["quota_path"], NOW) == 101` -> `assert 0 == 101`** in
`tests/test_youtube.py::test_every_request_is_logged_with_its_units`: the first full test run of this phase, before
any of my code ran there. A Phase 5 test that started failing on its own: `search_trends(now=...)` takes a clock
for testing, but `_get()` logged each quota line with the wall clock, and the test counts lines on `NOW`'s Pacific
day (2026-10-23). It passed while the real date was that day; on 2026-10-24 (05:50 in California) it failed.
Fix: `_get()` logs at the search's own `now` (in real use the same as the wall clock). Lesson: a test with a fixed
clock must not let the code under test read the real one anywhere.

**E2. `shortsmith make coffee --simple --max-rounds 1` exited 0** (the test expected a usage error). The check was
`ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE`. A debug script showed
`get_parameter_source` really returned `ParameterSource.COMMANDLINE`; `ls .venv/.../typer` showed a `_click`
folder: **Typer 0.27 vendors its own copy of Click** (`typer._click`), and the `click` package in the venv
(8.5.0, there for MLflow/Flask/uvicorn) is a different module, so its enum never equals Typer's. The same bug was
waiting in `failures()`: it re-raised `click.ClickException` so usage errors keep exit code 2, but a Typer
`BadParameter` isn't a `click.ClickException` and would have become "error ..., exit 1". Fixes: compare the source
by name (`.name == "COMMANDLINE"`); find Typer's own ClickException through `typer.BadParameter.__mro__`; `import
click` removed from cli.py. Lesson: when a library vendors a dependency, the "same" classes aren't the same.

**E3. `--RUN  RUN` in `cost --help`** (Rich help): an option `--run` with `metavar="RUN"` was printed as `--RUN`.
Found by a help test asserting `--run` in the output. Typer's Rich formatting seems to replace the option text when
the metavar equals the upper-cased name; I didn't chase it into Typer. Fix: `metavar="ID"`.

**E4. A brand-new run stopped before its first node: `spent $0.0662 + writer ~$0.0063 > budget $0.0100`.** Live
run 2's first attempt; the ledger had no line with that run id. $0.0662 was the sum of every ledger line WITHOUT a
`run_id` key (Phases 1-5). Cause: `LedgerEntry.run_id = Field(default_factory=CURRENT_RUN.get)`, and Pydantic runs a
default factory when it *parses* a JSON line that lacks the field, not only when code creates an entry. Inside a
node `CURRENT_RUN` is the current run, so every old line read back as this run's. Phase 6 never read the ledger
inside a node; the budget check is the first code that does. Phase 6's numbers were right (the live run's "11 calls
(11 in this session)" matches: LangGraph runs a node in a copied context, so the variable doesn't leak out to the
CLI's own reads). Fix: `ledger.entries()` parses with `CURRENT_RUN` set to None and resets it after; new lines are
still tagged when they are created. Regression test in test_ledger.py. Lesson: a context-dependent default is a
hidden input to *parsing*.

**E5. A run still running in the other window showed as `crashed at critic`.** Found by running `shortsmith list`
during live run 1. A checkpoint can't tell "a process is working on the next node" from "the process died before
finishing it": both are `next = critic`, no error. Risky, not just confusing: `resume` on it would start a second
process on the same thread, and both would pay for the critic. Fix: the `.running` marker with the process id
(§1); `list` says "running (critic)", `resume` refuses ("is running right now (process N)"). Test: a real sleeping
subprocess's pid in the marker -> "running", killed -> back to the checkpoint's own status.

**E6. PowerShell 5.1 garbled redirected Uzbek text** (`ΓÇö o╩╗zgacha`, `Γÿò`): §3. Found by checking the redirected
file's first bytes in the owner's own terminal (my build shell has UTF-8 defaults and looked fine, which would have
hidden it). Fix: `-o/--output FILE` on list/show/cost (UTF-8, no BOM); `--json` is ASCII-only by design.

**E7. Markdown tables, twice.** (a) Lines of only spaces above and below every table in a file: Rich's
`box.MARKDOWN` has a top and bottom edge made of spaces. Fix: in plain mode `Ui.print` captures the output and
drops whitespace-only lines and trailing spaces. (b) Columns shifted in `cost`'s "By model and step" and in
`doctor`: a cell held a `|` ("gemini-3.5-flash-lite | writer", "set | LLM calls ..."), which is the Markdown column
separator. Fix: model and step are two columns; doctor's details use `;`.

**E8. `--plain` in a real terminal wrapped every table row.** Plain mode had a fixed width of 200 columns (right for
a file or pipe, where there is no width) on a ~120-column screen. Found with `show --plain` in the terminal. Fix:
`--plain` on a terminal keeps the terminal's width; only a file or pipe gets 200.

**E9. `2026/10/24 18:37:28 INFO mlflow.tracing.export.async_export_queue: Flushing the async trace logging queue
before program exit. This may take a while...`** at the end of every traced command. Fix: `tracing.setup()` sets
`MLFLOW_LOGGING_LEVEL=WARNING` (found in mlflow/utils/logging_utils.py).

**E10. `shortsmith mcp` logged nothing on stderr**, where `python -m shortsmith.mcp_server` logs its startup line.
The CLI's callback sets the `shortsmith` logger to WARNING (so warnings print and INFO doesn't), and the server's
logger is `shortsmith.mcp`, a child. Fix: the `mcp` command gives the logger back (level NOTSET, the CLI's
handler removed) before starting the server, which configures its own log.

**E11. DailyQuotaError's message wasn't shortened.** `explain()` split it on `": try another model"`; the text says
`"; try another model"`. Found by its new test.

**E12 (test bugs, found by the tests).** (a) I expected 7 LLM calls for the fake graph run; it's 6 (writer 2,
director 1, critic 2, revise 1), as Phase 6 E4(a) already said. (b) I expected `resume --budget 0.02`; it suggests
0.04 because the suggestion covers every remaining round (3 critic + 2 revise estimates), by design. (c) Markdown
cells are padded: `"| 8 "` vs `"|     8 |"` (a regex now). (d) A paused run has no script.json yet, so rerun-scene's
"no script.json" check fired before the more useful "not finished (paused at approval)": the status check now
comes first (a real UX improvement the test pushed). (e) Twice, a Python heredoc's `\\n` / `\\\\u02bb` lost a level
of escaping on the way into a file: once a real line break inside a string in cli.py (`SyntaxError: unterminated
string literal`), once a test comparing against the real `ʻ` instead of the escape `\u02bb`. Edited by hand.

**E13. Not fixed, not reproduced: a block of ~25 blank lines** once in the app's terminal panel, between the editor's
spinner and its `done` line in live run 1 (rounds 0 and 1 had none). Rich's transient progress moves the cursor up
to erase itself; a panel resize while it runs could leave this. Noted, not chased.

**Findings that were not bugs.** LangGraph keeps a node's exception in the checkpoint as its repr string (with
surrounding quotes), not as the exception object; `update_state()` with no `as_node` keeps `next` right after a
stop before writer, director or critic and while paused at an interrupt (it drops the pending interrupt, and the
node asks again when it re-runs). Both checked in a scratch prototype before the budget design relied on them.

---

## 5. Design decisions and why

**D1. Typer (with Rich), not argparse or plain Click.** The commands are functions whose type hints become the
options, checks and help (`int` with `min=0`, an `Enum` for `--lang`, `bool | None` for `--mock/--real`, `Path` with
`exists=True`). argparse (make.py) needs every option spelled twice (parser + use) and has no command groups or
env-var support; Click is what Typer wraps (and vendors, E2). Typer was already in the lock (mcp[cli]), so it cost
no new package. What it cost: E2 and E3, and Rich-formatted help that needs `rich_markup_mode=None` to be plain
when piped.

**D2. A new `shortsmith` command, and `python -m shortsmith.make` left as it was** (not turned into a thin wrapper).
The lessons, Phase 6's CLI test and its output strings use make.py's argparse interface (`--graph`, `--resume`,
`--run FOLDER`); a wrapper translating it to the new commands would change its output and break them, for no gain.
The new command reuses make.py's functions for `--simple`, and graph.py's `describe()` is shared, so nothing is
written twice.

**D3. stdout for results, stderr for progress and messages.** `shortsmith make ... > summary.txt` still shows
progress on screen; `list --json` is clean JSON on stdout (tested by `json.loads(result.stdout)` with nothing
else); a script can pipe results without parsing around spinners. Prompts (the hero question, `--real`) go to
stderr too.

**D4. Rich in a terminal, plain lines otherwise, decided per stream.** stdout piped + stderr a terminal = plain
tables in the file, a live spinner on screen. Plain = no ANSI codes (tested on every command), no animation, a
`...` start line per node (a log shows what was running), Markdown tables (ASCII, readable in a file, render in
notes). Markup is off everywhere: run text like `[1, 3]` or `[en|uz|ru]` is printed as it is.

**D5. `--json` is ASCII-only** (`ensure_ascii=True`): it survives PowerShell 5.1's re-encoding (E6), any console code
page, and every JSON reader. The models are the same Pydantic models the tables come from (`RunSummary`,
`RunDetail`, `CostReport`), so Phase 8's API can return them as they are.

**D6. The budget lives in the graph, not the CLI.** The node wrapper checks it, so any front end (Phase 8's web API)
gets the same hard stop. It raises before the node's work, so the stop is a normal LangGraph failure: the
checkpoint of the last finished node stays, the error is stored, resume re-runs the node. Settings that must
survive a resume (`budget_usd`, `media`, `hero_clips`) are RunState fields; `resume --budget/--mock/--real` writes
them with `update_state()`, so a later resume without flags keeps them. A real-media run resumed without `--real`
stays real (asked again), and a mock run never turns real because of the environment.

**D7. Conservative estimates (the most expensive of the last 10 similar calls).** A budget is a limit: an estimate
that is usually a bit high stops a run slightly early; an average-based one overshoots about half the time. It also
follows prompt changes (a longer prompt raises its own history) without anyone updating a table.

**D8. Exit codes with fixed meanings, and the "paused" and "over budget" outcomes as their own codes.** A script
(or Phase 9's daily job) can tell "needs a person" (3) from "needs money" (4) from "broken" (1) without reading text.
2 stays Click's usage code. `resume` of a finished run is 0 ("nothing to do"), so resuming is idempotent.

**D9. Errors as one line + `next:`, a traceback only with `--debug`.** `explain()` maps the errors we know (the
provider refusing the key, a 429 after waiting, the daily quota, no connection, ffmpeg missing, a model answer
breaking a rule, an unknown run) to a sentence and the next step; anything else is `Type: message` plus the resume
command and "run again with --debug". Typer's pretty exceptions are off: they print local variables by default,
and `llm._client()` has the API key in one.

**D10. `.env` loaded by the command** (`override=False`). `uv run shortsmith ...` works without `--env-file` (the
space in the repo path made absolute `--env-file` paths a known gotcha); the real environment still wins.

**D11. `--real` is never silent.** An estimate (images, the hero clip asked again before it is made, revision images
at most, LLM list price billed $0), then a yes/no on stderr, or `--yes`; with nobody to ask and no `--yes` it is a
usage error before any folder or call. `FAL_KEY` is checked first. Never run for real in this phase.

**D12. rerun-scene goes through the producer's manifest, `only={scene}`.** The same rules decide what is stale, the
same archive step keeps the old file, and the other scenes are never looked at (so a missing file elsewhere isn't
silently made and paid for). No LLM: the person writes the words. The checkpoint isn't changed: an `update_state`
on a finished run would recompute `next` from the critic's edge and could make the run look resumable.

**D13. Status from the checkpoint, plus one marker.** No status file: it would be a second truth that a crash, the
legacy `make --resume` or a manual resume can leave stale. The only fact the checkpoint lacks, "is a process still
on it", comes from `.running` + a liveness check (E5).

**D14. Defaults chosen**: graph pipeline (the Phase 6 work is the product); mock media (a forgotten setting must
never cost money, as in images.py); `--hero-clips 1` (the video target); no budget (but every run shows its cost);
`show` defaults to `latest`; `list` shows 20.

**D15. `doctor` makes no network call.** A "does the key work" call would spend one of the day's 20 Flash requests
or the YouTube quota; set/not set + the provider + ffmpeg + the stores catches the usual setup problems for free.

---

## 6. CLI usability: trying it as a new user, from `--help` only

What I tried, in order, reading nothing but help screens, and what I changed:

1. `shortsmith` alone -> the help (`no_args_is_help`). OK.
2. The command list cut descriptions mid-sentence in plain mode (`make  Make a video from a topic: research
   (optional), writer,...`). Gave every command a one-line `short_help`; grouped them (Make videos / Look at runs /
   Setup).
3. `make --help`: `--research / --no-research [default: no-research]`, `--mcp / --no-mcp`, `--simple /
   --no-simple` were noise for on/off switches. Now single flags. `<str>`, `<int range>`, `<float range>` said
   nothing: metavars `NAME`, `N`, `SCORE`, `MODEL`, `FILE`, `TEXT`, `USD`. Options grouped in Rich help (Money,
   Critic loop, Advanced).
4. "What does --budget count?" The help says LIST price, the header says it again with the free-tier note.
5. `make` without a topic -> `Missing argument 'TOPIC'` (exit 2). `--hero-clips 2` -> explains why only 0 or 1.
6. "Which run?": every RUN argument takes `latest`, a unique prefix or a path; `show` defaults to latest; an
   ambiguous prefix lists the matches.
7. `--debug` typed at the end failed with "No such option" before `main()` moved it to the front.
8. A long free-tier wait looked frozen: the spinner shows elapsed time and the node's estimate.
9. `list` during a run said "crashed" (E5) -> "running (critic)".
10. Redirecting `show` in PowerShell garbled Uzbek (E6) -> `-o FILE`, documented in the option's help.
11. `--plain` in a terminal wrapped tables (E8); piped tables had whitespace lines and a broken `|` column (E7).
12. `rerun-scene` on a paused run said "no script.json" before saying the useful thing (E12d).
13. Left as is: Typer's plain help writes `{TOPIC}` / `{RUN}` in the usage line; `approve_hero` shows two start lines
    around its question (it really runs twice); the header line wraps on a narrow terminal.

**Slow start, measured, not fixed**: `shortsmith list` takes ~5-7 s on this machine (10.7 s the first time).
`import shortsmith.runs` is 3.5 s cold (openai 1.4 s, edge_tts 0.6 s, httpx 0.3 s: the run models import llm.py
and the media modules) and opening the checkpoints imports LangGraph (1.9 s); the listing itself takes 0.05 s.
Making edge_tts lazy would save 0.6 s but means rewriting Phase 2's tts tests (they patch `tts.edge_tts`); not worth
it now. `cost` and `doctor` don't open the checkpoints.

---

## 7. Tests

`uv run pytest`: **391 passed** (323 + 68), ~70 s (the CLI tests run the real LangGraph + SQLite many times; one
test starts `python -m shortsmith` and `python -m shortsmith.make` as subprocesses). New: test_cli **43**, test_budget
**16**, test_rerun **8**, test_ledger **+1**. All offline, with Phase 6's fakes (`from test_graph import world`): a
fake OpenAI client answering by schema name, fake edge-tts, ffmpeg assembly, frames and ffprobe, a fake
`shutil.which` for ffmpeg, a fake key; real LangGraph and SQLite checkpoints in a temp folder; Typer's own
`CliRunner` (stdout and stderr separate).
- **test_cli.py**: the help tree, exit codes and settings order; every command's help lists its flags (and hides
  `--crash-before`); make -> final.mp4 (results on stdout, progress on stderr, a Markdown table, no ANSI, the marker
  removed); no answer -> exit 3 -> `resume --approve no` -> Ken Burns -> resume again "already finished" (exit 0);
  **the budget stop** with a fixture ledger: exit 4 before the critic, `next == ("critic",)`, the suggested budget,
  ledger steps writer/writer-pick/director only, `list` status, then `resume --budget 0.05` finishes without a new
  writer call and saves the budget; SHORTSMITH_BUDGET from the environment stops a run, `--budget 1` overrides it;
  `--hero-clips 0` (no question, no clip, no ai_clip); `--hero-clips 2` (exit 2, no call); **`--real`** refused
  without `--yes` and no terminal (exit 2, no folder, no call, the estimate lines), declined at a simulated prompt
  (exit 1), with `--yes` (fake media) recorded as `media: real` and the producer's estimate $0.0180; `--mock` beats
  `MOCK_MEDIA=0`; a missing key fails before any folder; a failing node = one message + resume hint, "crashed at
  director", and `--debug` lets the exception through; a running process's marker -> "running", resume refused;
  Ctrl+C -> 130; the quota message; global flags after the command; `--simple` (tagged ledger line, "no
  checkpoint", a graph-only flag refused); `list` order/status/JSON; `show` sections and JSON; `latest` and prefixes;
  an unknown run; `cost` tables and JSON (`--run`); **rerun-scene** (only scene 3's image, the archive, the style
  words, the voice untouched, no ledger line, reruns.json, `show` lists it; narration -> voice only; the hero
  picture asks and pauses with nothing done, "no" -> pan_left and the clip archived, "yes" -> a new clip; an
  unfinished run and scene 9 refused; a straight run adopted); `-o FILE` writes strict UTF-8 without BOM and JSON is
  ASCII-escaped; doctor never prints a key and fails without the provider key; no colour codes with NO_COLOR and
  `--plain`; `python -m shortsmith.make --help` and `python -m shortsmith --help`.
- **test_budget.py**: the max of the last 10, the same step on the same model (dated names match), the first-call
  guess, writer = options + pick, the critic on the critic model, free nodes, the producer free with mock media and
  priced with real media, the stop rule's numbers and message, a free node and no budget never stop,
  `nodes_left()`, the whole-run estimate; costs: billed $0 for Gemini on the free tier and list for OpenAI/fal,
  `GEMINI_TIER=paid`, the three groupings, the filters, the quota day starting at 07:00 UTC; `status_of` for every
  kind (with LangGraph's quoted repr strings).
- **test_rerun.py**: `plan_change` (no option = both, the style words, nothing to redo unless forced, a new hook, a
  too-long hook refused, a new CTA, the hero question only when the approved hero picture changes, a missing scene)
  and `resolve`.
- `uv run pytest exercises/phase-7`: 18 fail with NotImplementedError (by design); the same 18 pass against a
  reference solution (checked in a scratch folder, not committed). `uv run pytest exercises`: 82 fail (64 + 18), no
  module-name clashes.

---

## 8. What it cost

From `runs/ledger.jsonl`, everything this phase sent (2026-10-24, 13:24-13:52 UTC; list prices; **billed $0**):

| run | step | model | calls | input tokens | output tokens (incl. thinking) | list price |
|---|---|---|---:|---:|---:|---:|
| 1 (tea houses, uz) | researcher | gemini-3.5-flash-lite | 1 | 1,522 | 298 | $0.001202 |
| 1 | rag (embedding) | gemini-embedding-2 | 1 | 16 | 0 | $0.000003 |
| 1 | writer (2 attempts) | gemini-3.5-flash-lite | 2 | 1,544 | 3,239 | $0.008561 |
| 1 | writer-pick | gemini-3.5-flash-lite | 1 | 811 | 163 | $0.000651 |
| 1 | director | gemini-3.5-flash-lite | 1 | 750 | 505 | $0.001488 |
| 1 | critic | gemini-3.8-flash | 3 | 16,172 | 2,227 | $0.020480 |
| 1 | revise | gemini-3.5-flash-lite | 2 | 1,626 | 1,042 | $0.003093 |
| 2 (metro, en) | writer | gemini-3.5-flash-lite | 1 | 517 | 1,893 | $0.004888 |
| 2 | writer-pick | gemini-3.5-flash-lite | 1 | 645 | 171 | $0.000621 |
| 2 | director | gemini-3.5-flash-lite | 1 | 700 | 566 | $0.001625 |
| 2 | critic | gemini-3.8-flash | 1 | 5,389 | 500 | $0.005917 |
| **total** | | | **15** | **29,692** | **10,604** | **$0.048527** |

- **14 LLM calls + 1 embedding request** (the plan's ceiling was ~40). gemini-3.8-flash: 4 of its 20 requests today.
  Thinking tokens (all from 3.8 Flash's critic): 2,100 of the 10,604 output tokens.
- YouTube: two live searches, 202 units of our 2,000 cap (run 1's research: `"tea houses of Samarkand"` found fewer
  than 3 Shorts, so Phase 5's `broader()` tried `"tea houses of"`: it drops the LAST word, here "Samarkand", the one
  that mattered. Open item below).
- Media $0: placeholders, two mock hero clips, edge-tts, ffmpeg. `rerun-scene` added no line.
- Ledger total for the whole project: $0.2633 list price, $0 billed for Gemini.
- Wall time: run 1 761 s, of which the critic 567 s (135 + 165 + 266 s in the free tier's queue); run 2 132 s + 133 s.

---

## 9. Open items

- **Start-up time** (~5-7 s for `list`/`show`): §6. Lazy imports in llm.py (openai) and tts.py (edge_tts) would help
  every read-only command; both need their tests adjusted.
- **The writer's retry isn't in the estimate.** The estimate could add one options call with the probability seen
  in the ledger (Uzbek: often), or the budget could treat the writer as 1-2 options calls.
- **A live interactive prompt** (typing y/N at the hero question or the `--real` question) was tested only with a
  simulated terminal (`stdin_interactive` patched); the terminal tool can't type into a running program. Worth one
  manual try by the owner: `uv run shortsmith make "topic"` and answer the question.
- **`--real`** is wired, estimated and guarded, but never run (no credits, and not allowed in this phase).
- **More than one hero clip** needs the director to choose scenes, an approval per clip and a price check per clip.
- **YouTube's broader query** drops the last word of the topic ("tea houses of Samarkand" -> "tea houses of"); it
  should drop filler first, or keep the place name.
- **Videos run long in English** (run 2: 34.4 s, 12-13 words per scene) and the critic loop still doesn't converge
  (run 1: 5.5 -> 6 -> 6); both are Phase 6's open items, seen again.
- **Ctrl+C mid-run** (exit 130 + resume hint) is tested through `explain()`, not by interrupting a live run.
- **Trace ids** come from MLflow's internal tables; a new MLflow version could rename them (only the ids would go
  missing, with a warning).
- The Claude Desktop test from Phase 4 is still the owner's; `shortsmith mcp` is an alternative command for it.
