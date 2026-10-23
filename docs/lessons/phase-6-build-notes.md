# Phase 6 build notes (Steps 6.1-6.9: LangGraph orchestration + MLflow)

Written by the build session for the teaching session. Everything below happened on this machine
(Windows 11 Home with Smart App Control on, Python 3.12, uv 0.13.1, **langgraph 1.2.16**,
langgraph-checkpoint 4.3.0, langgraph-checkpoint-sqlite 3.1.2, langchain-core 1.7.0, **mlflow 3.17.2**,
openai 3.33.0, pydantic 2.14.1) on 2026-10-23/24 (UTC evening, Tashkent night). Money spent: **$0**
(Gemini free tier for every LLM, vision and embedding call; placeholder images; mock hero clips; edge-tts;
local ffmpeg; YouTube within our cap).

**Done-check, in one paragraph.** `make "topic" --graph` runs the whole pipeline as a LangGraph graph:
researcher (optional) -> writer (3 options + checks + pick) -> director (shots, style, hero scene) ->
approve_hero (an `interrupt` before the paid clip) -> producer (plain code, media through the tool registry
or the MCP server, make-like) -> editor (ffmpeg) -> critic (4 frames + the script, one vision call) ->
revise -> back to approve_hero, at most 2 revision rounds. State is a Pydantic `RunState` with reducers
(append for history, a custom per-node sum for usage), checkpointed in SQLite after every node
(`durability="sync"`, `thread_id = run_id`), so `make --resume <run_id>` continues a run. **Run 1** (coffee
shop, en) went round the loop twice and stopped at max rounds (scores 5.5 -> 6.2 -> 6.0; scenes 1, 3, 5
revised, then 1), surviving a real 503 from Gemini by resuming; **run 2** (Registan, uz) was crashed on
purpose before the director, resumed into a paused approval question, answered "no" (Ken Burns for the
hero scene), killed for real with `Stop-Process -Force` mid-producer (a 0-byte voice file left behind and
correctly not trusted), resumed, and passed its review (7.8); **run 3** (saving money, en) ran in one go
with the producer going through the MCP server and is one complete 15-span MLflow trace. No resume paid for
anything already done (ledger lines before/after in §5). The critic's vision path was proven on a real
photograph (it described "an outdoor sunset over water" and flagged the mismatch). The **v1/v2 evaluation**
(5 topics, en + uz, blind stronger judge): mean critic score **7.40 vs 7.40**, but v2 removed copied hooks
(**11/14 options -> 0/14**) and put more options inside the word target (**2/14 -> 7/14**). **Context
trimming**: the four flash-lite nodes read **2,926** prompt tokens instead of **26,109** with the whole state
(6-10x per node). 64 LLM calls in the phase ($0.149 list price, $0 billed). **323 tests pass** (235 before);
the exercise `exercises/phase-6/` has 14 tests that fail with NotImplementedError by design.

**Three things the lesson should not miss**, all found by running it, not by planning it: (1) Gemini's
free tier gives each Flash model **20 requests a day**: 3.8 Flash ran out mid-phase and the critic moved to
3.6 Flash (E11); (2) Gemini's compatible endpoint **leaves thinking tokens out of `completion_tokens`**, so
every Flash call in the ledger was under-priced until this phase fixed it (E13: for the 3.6 Flash critic the
thinking was 5x the visible output); (3) in the evaluator-optimizer loop the **reviser invented a business
name** to satisfy the critic, and the critic then **accepted it as a fact** (E6, §3).

**Plan changes to make (teaching session's files):** PLAN §5's `CriticReport.reasons: dict[int, str]` is a
list of `SceneIssue(scene, problem, reason, fix)` (a strict JSON schema can't express integer keys), and
`RunState.tokens_used/cost_usd` are now derived from a per-node `usage` map. PLAN §4's tree: the graph
lives in `agents/graph.py` with `agents/state.py`, `writer.py`, `director.py`, `producer.py`, `critic.py`,
`reviser.py`; there is no `editor.py` (the editor is a 10-line node). PLAN §3: the critic's judge is
`gemini-3.8-flash` with `gemini-3.6-flash` as the daily-quota fallback (`SHORTSMITH_CRITIC_MODEL`); MLflow
3.17's default store is `sqlite:///mlflow.db` (not `./mlruns`). CLAUDE.md could note: the 20-requests-a-day
limit per Flash model, Smart App Control blocking fresh native wheels (pins in `[tool.uv]`), and the new
`make --graph / --resume / --approve` flags.

---

## 1. What was built

### Files

| File | Status | What it is |
|---|---|---|
| `pyproject.toml`, `uv.lock` | changed | `uv add langgraph langgraph-checkpoint-sqlite mlflow`: langgraph 1.2.16, langgraph-checkpoint 4.3.0, langgraph-checkpoint-sqlite 3.1.2, **langchain-core 1.7.0** (a hard dependency of langgraph: it uses langchain-core's `Runnable`/config/callbacks plumbing; we import no LangChain model wrapper), mlflow 3.17.2 (+ mlflow-skinny, mlflow-tracing; 50 packages: pandas, pyarrow, scikit-learn, scipy, sqlalchemy, alembic, flask, matplotlib, ...; a 13-minute download). `[tool.uv] constraint-dependencies` pins three versions whose DLLs Windows Smart App Control lets load (E1). python-dateutil went from 3.0.0 back to 2.9.0.post0 in the resolution. |
| `.gitignore` | changed | `checkpoints/`, `*.sqlite-*`, `*.db-*` (SQLite's WAL and shared-memory files) next to the existing `*.sqlite`, `*.db`, `mlruns/` |
| `src/shortsmith/agents/toy_graph.py` | new | 6.1: 3 nodes, 1 conditional edge, a reducer |
| `src/shortsmith/agents/state.py` | new | `RunState`, `Usage`, the `add_usage` reducer |
| `src/shortsmith/agents/writer.py` | new | 3 options in one call, checked one by one, a pick call; prompt versions `v1`/`v2`; copied-hook detection |
| `src/shortsmith/agents/director.py` | new | the shot list: style, image prompts, camera moves, the hero scene (`HeroPlan`) |
| `src/shortsmith/agents/producer.py` | new | deterministic producer with a make-like manifest; direct or MCP tool calls |
| `src/shortsmith/agents/critic.py` | new | frame sampling (ffmpeg), the vision request (base64 data URLs), code checks, `review()` |
| `src/shortsmith/agents/reviser.py` | new | the per-scene revision between critic and producer |
| `src/shortsmith/agents/graph.py` | new | node wrapper, nodes, edges, SQLite checkpointer, interrupt, `drive()`, the CLI driver |
| `src/shortsmith/agents/prompt_eval.py` | new | 6.8: v1 vs v2 on 5 topics, MLflow runs |
| `src/shortsmith/agents/context_eval.py` | new | 6.9: trimmed vs whole-state requests, rebuilt from checkpoints |
| `src/shortsmith/tracing.py` | new | 6.7: MLflow setup, safe spans, image shrinking, MLflow runs |
| `src/shortsmith/media/animate.py` | new | the hero clip: mock clip (ffmpeg) or fal image-to-video (never run) |
| `src/shortsmith/agents/tools.py` | changed | `animate_image` tool in a new `"clip"` group (`ClipArgs`, `ClipResult`) |
| `src/shortsmith/mcp_server.py` | changed | media **and clip** tools get a `run_id`; the instructions mention the paid clip |
| `src/shortsmith/media/ffmpeg.py` | changed | `AI_CLIPS`, `ai_clip_file()`, `clip_cmd()`; `assemble()` uses a hero clip when the scene's motion is `ai_clip` and the file exists |
| `src/shortsmith/llm.py` | changed | `chat_schema()` (structured output without validation; `chat_json` now calls it), `paced()` (one request per 4.5 s, 429 waits as long as Gemini says, via the existing `_retry_delay`) |
| `src/shortsmith/ledger.py` | changed | `CURRENT_RUN` context variable, `LedgerEntry.run_id`, `entries(path, run_id)`; hero-clip price estimate in `MEDIA_PRICES` |
| `src/shortsmith/models.py` | changed | `SceneIssue`, `CriticVerdict`, `ScriptChecks`, `CriticReport`, `CRITIC_MAX_WEAK` |
| `src/shortsmith/make.py` | changed | `--graph`, `--resume`, `--approve`, `--max-rounds`, `--pass-score`, `--prompt`, `--mcp`, `--no-trace`, `--crash-before`; the old pipeline is untouched |
| `tests/test_graph.py` (26), `test_critic.py` (23), `test_producer.py` (14), `test_writer.py` (12), `test_toy_graph.py` (4) | new | offline (§13) |
| `tests/test_llm.py` (+7), `test_ledger.py` (+2), `test_mcp_server.py` (3 changed: the server lists 8 tools now) | changed | |
| `exercises/phase-6/graph_runner.py` + `test_graph_runner.py` | new | 🧠 `merge_update`, `route_after_critic`, `run_graph`; 14 tests |

### The graph

```
                 research_on?
  START ──────────┬─────────────> researcher ──┐
                  └───────────────────────────>├─> writer ──> director ──> approve_hero ──> producer ──> editor ──> critic
                                                                               ▲                                     │
                                                                               │        weak scenes, score < 7,      │
                                                                               └──────── revise <── rounds left ─────┤
                                                                                                                     │ otherwise
                                                                                                                     ▼
                                                                                                                    END
```
`build()` wires it: one conditional edge from START (`route_start`), fixed edges for the known steps, one
conditional edge after the critic (`route_after_critic`), and `revise -> approve_hero` (a revision that
changes the hero image needs a new paid clip, so it is asked again). `get_graph().draw_mermaid()` prints it.

| node | LLM calls | what it does | returns (fields it changes) |
|---|---|---|---|
| researcher | 1 (+1-2 embedding requests, YouTube cached) | Phase 5's `research()` as it is; saves research.json | `research` (ResearchNotes) |
| writer | 1 + 0-1 | 3 options in one structured call, each validated alone, one pick call; saves options.json | `script_options`, `rejected_options`, `pick`, `chosen` |
| director | 1 | shots, style, hero scene + motion; code applies them; saves director.json | `plan`, `hero` |
| approve_hero | 0 | `interrupt()` before a new paid clip; "no" -> Ken Burns for that scene | `hero.approved`, (`plan`) |
| producer | 0 | images, voices (+ the approved clip) for whatever is missing or out of date | `made` |
| editor | 0 | `ffmpeg.assemble()` (a hero clip is used for its scene); keeps the replaced final.mp4 | `final_path` |
| critic | 1 (vision) | 4 frames + script + code facts -> `CriticReport`; saves critic/round_N/ | `critic_rounds` (appended) |
| revise | 1 | the flagged scenes rewritten (only the fields their problem names) | `plan`, `revisions` (appended) |

Every node is wrapped by `node(name)`: sets `ledger.CURRENT_RUN`, can be crashed on purpose
(`SHORTSMITH_CRASH_BEFORE=<node>` -> `os._exit(70)`), opens a tracing span with what the node reads
(`reads()`) and a summary of what it returned (`brief()`), adds the node's wall time to its usage line, and
closes the span normally when the node pauses on an interrupt (it's a pause, not a failure).

### `RunState` (agents/state.py)

| field | type | reducer | set by |
|---|---|---|---|
| run_id, run_dir, topic, language, brand | str ... | replace | `new_state()` (once) |
| research_on, prompt_version, max_rounds, pass_score, via_mcp | settings | replace | CLI |
| research | ResearchNotes \| None | replace | researcher |
| script_options, rejected_options, pick, chosen | list[Script], list[str], WriterPick, Script | replace | writer |
| plan | Script | replace | director, approve_hero (on "no"), revise |
| hero | HeroPlan (scene, motion_prompt, reason, fallback, cost_usd, approved) | replace | director, approve_hero |
| final_path | str | replace | editor |
| made | list[str] | **operator.add** | producer ("round 1: voices 1, 3, 5") |
| critic_rounds | list[CriticReport] | **operator.add** | critic |
| revisions | list[Revision] (round, scene, problem, reason, fix, before/after) | **operator.add** | revise |
| usage | dict[str, Usage] (calls, prompt_tokens, completion_tokens, cost_usd, seconds) | **add_usage** (sum per node) | every node |

Properties: `round` (= `len(critic_rounds)`), `tokens_used`, `cost_usd` (PLAN §5's totals, derived).
`test_the_state_without_reducers_would_lose_history` checks LangGraph made the right channels:
`BinaryOperatorAggregate` for critic_rounds and usage, `LastValue` for plan.

### The run folder after a graph run

`script.json` (the current plan), `options.json`, `director.json`, `research.json`, `media.json` (the
producer's manifest), `scenes/`, `audio/`, `clips_ai/scene_01.mp4` (the hero clip), `clips/`, `final.mp4`,
`critic/round_N/{scene_NN.jpg, report.json}`, `history/round_N/` (what round N replaced, final.mp4 included).
`make --run <folder>` (the old path) still rebuilds final.mp4 from such a folder.

---

## 2. The toy graph (6.1)

`uv run python -m shortsmith.agents.toy_graph` (agents/toy_graph.py): a Pydantic `ToyState(topic, hook,
words, checks, history: Annotated[list[str], operator.add])`, nodes `write -> check -(too long?)-> shorten ->
check ...`, 13 -> 11 -> 9 -> 7 words. Output, abridged:

```
--- stream(stream_mode='updates'): what each node returned ---
{'write': {'hook': 'Here is everything you never knew you needed to know about plov today', 'history': ['write: 13 words']}}
{'check': {'words': 13, 'checks': 1, 'history': ['check: 13 words']}}
{'shorten': {'hook': 'everything you never knew you needed to know about plov today', 'history': ["shorten: -> '...'"]}}
... (check 11, shorten, check 9, shorten)
{'check': {'words': 7, 'checks': 4, 'history': ['check: 7 words']}}

--- stream(stream_mode='values'): the whole state after each step ---
0 {'topic': 'plov'} history: 0 lines
1 {'topic': 'plov', 'hook': 'Here is everything you never knew ... plov today'} history: 1 lines
2 {'topic': 'plov', 'hook': 'Here is ... plov today', 'words': 13, 'checks': 1} history: 2 lines
...
8 {'topic': 'plov', 'hook': 'you needed to know about plov today', 'words': 7, 'checks': 4} history: 8 lines

--- invoke(): the final state ---
dict {'topic': 'plov', 'hook': 'you needed to know about plov today', 'words': 7, 'checks': 4, 'history': [8 lines]}
```
What it shows: (1) a node returns only what it changed ("updates"); (2) `hook` is replaced each time, while
`history` (with `operator.add`) keeps every line; (3) **`invoke()` returns a plain `dict`, not a `ToyState`**,
even with a Pydantic state schema; (4) fields never written are absent (step 0 has no `hook`, though the
model's default is ""): defaults aren't channel values; (5) the conditional edge is a function of the state
that only names the next node. `draw_mermaid()` prints the graph (`draw_ascii()` needs grandalf, E2).

Before trusting tutorials I probed the installed version (scratch `probe_lg.py`): a Pydantic state with
nested models round-trips through `SqliteSaver` into a new process; an `interrupt()` makes `invoke()` return
the state plus `'__interrupt__': [Interrupt(value=..., id=...)]`; `Command(resume="yes")` in a new process
re-runs only the interrupted node; a node that raises leaves `tasks[0].error` in the checkpoint and resuming
re-runs only that node; single-task supersteps run in the main thread (context variables are visible). The
serializer warns that by default any class named in a checkpoint is imported on load (D11).

---

## 3. Run 1: topic -> final.mp4 through the whole graph, with the critic loop to max rounds

`make "coffee shop in Tashkent, cozy autumn vibe" --graph --research --approve yes` -> `runs/2026-10-23_233023`
(mock images, Gemini free tier). The first invocation ran untraced (MLflow failed to import, E1) and died in
the third review on a 503 (§5); `make --resume 2026-10-23_233023` finished it. Console, both parts:

```
run 2026-10-23_233023 | 'coffee shop in Tashkent, cozy autumn vibe' (en) | prompt v2 | max 2 revisions, pass at 7 | images: mock placeholders
researcher       4.3 s |   1477 in   349 out | notes: 3 hook patterns, 3 angles
writer          18.3 s |   1486 in  2183 out | 3 options, picked 1: 'Stop scrolling for the best Tashkent spot.'
director         2.4 s |    685 in   491 out | hero = scene 1: steam curls up from the surface of the warm latte as the camera slowly pushes in.
?? Scene 1 ("Stop scrolling for the best Tashkent spot. Autumn chills require warm ") as a 5-second AI clip (...) costs ~$0.30 with real media; MOCK_MEDIA=1 makes a free placeholder clip now. Continue?
   answered by --approve: yes
approve_hero     0.0 s |                     | hero clip approved
producer        76.7 s |                     | round 0: images 1, 2, 3, 4, 5, 6; voices 1, 2, 3, 4, 5, 6; clip 1
editor          29.3 s |                     | runs\2026-10-23_233023\final.mp4
critic         363.4 s |   5322 in   288 out | round 0: score 5.5, weak [5, 1, 3] (frames of scenes [1, 3, 4, 6])
revise           5.2 s |    755 in   477 out | scene 1 narration; scene 3 narration; scene 5 narration
approve_hero     0.0 s |                     | no new clip needed
producer        22.9 s |                     | round 1: voices 1, 3, 5
editor          24.7 s |                     | runs\2026-10-23_233023\final.mp4
critic         473.3 s |   5321 in   186 out | round 1: score 6.2, weak [1] (frames of scenes [1, 3, 4, 5])
revise           2.1 s |    683 in   394 out | scene 1 narration
approve_hero     0.0 s |                     | no new clip needed
producer        12.0 s |                     | round 2: voices 1
editor          22.6 s |                     | runs\2026-10-23_233023\final.mp4
   ... 503 in the critic: process exits (§5) ...
resuming 2026-10-23_233023 before critic ('coffee shop in Tashkent, cozy autumn vibe')
critic         500.9 s |   5318 in   213 out | round 2: score 6, weak [2, 5] (frames of scenes [1, 3, 4, 6])
done in 501 s | ledger lines for this run: 10 (1 new) | 25,645 tokens, $0.025811 list price | trace tr-5d611283ea3ccabc483f1d452dd7b5fc
stop: max rounds (2 revisions; last score 6) | scores by round: [5.5, 6.2, 6.0]

'Cozy Autumn Coffee in Tashkent' (en)
scene  motion      planned   voice  on screen  words
1      ai_clip        3.5s    3.5s       3.5s      9
2      pan_left       3.2s    6.6s       6.5s     14
3      pan_right      3.3s    5.1s       5.0s     13
4      zoom_out       3.4s    6.3s       6.3s     14
5      pan_right      3.2s    5.8s       5.7s     14
6      zoom_in        3.4s    5.7s       6.3s     13
total                20.0s   33.0s      33.3s     77
runs\2026-10-23_233023\final.mp4  1080x1920, 30 fps, 33.3 s, 4.1 MB, h264/yuv420p + aac
```

**Per node, the whole run** (RunState.usage, the summing reducer; seconds = wall time incl. free-tier queueing):

| node | runs | LLM calls | prompt tokens | completion tokens | list price | seconds |
|---|---:|---:|---:|---:|---:|---:|
| researcher | 1 | 1 | 1,477 | 349 | $0.001316 | 4.3 |
| writer | 1 | 2 (options + pick) | 1,486 | 2,183 | $0.005903 | 18.3 |
| director | 1 | 1 | 685 | 491 | $0.001433 | 2.4 |
| approve_hero | 3 (+1 paused) | 0 | | | | 0.0 |
| producer | 3 | 0 | | | | 111.6 |
| editor | 3 | 0 | | | | 76.5 |
| critic | 3 | 3 (vision, gemini-3.8-flash) | 15,961 | 687 | $0.014547 | 1,337.6 |
| revise | 2 | 2 | 1,438 | 871 | $0.002609 | 7.4 |
| **total** | | **10** | **21,047** | **4,581** | **$0.025808** | |

(The ledger's run total, $0.025811, adds the one embedding request of the research step.) The critic is
65% of the tokens (4 images of ~1,100 tokens each per review) and 86% of the wall time: 6-8 minutes per call
in the free tier's queue, while the same model answered in seconds earlier in the day.

**The three reviews and what changed** (frames: 4 per review; the scenes regenerated last time first):

| round | score | frames | words | weak scenes (problem: reason) |
|---|---:|---|---:|---|
| 0 | 5.5 | 1, 3, 4, 6 | 84 | **5** narration: "Claims to provide an exact address without naming the venue or street." · **1** narration: "overused cliché instead of highlighting a unique feature" · **3** narration: "Phrasing like 'happy mouth' feels unnatural" |
| 1 | 6.2 | 1, 3, 4, **5** | 80 | **1** narration: "somewhat generic and takes too long to state the specific location or payoff"; fix: *Start directly with the unique location: 'Is this the coziest autumn coffee spot in Tashkent?'* |
| 2 | 6.0 | 1, 3, 4, 6 | 77 | **2** narration: "Broadening to Central Asia dilutes the focus" · **5** narration: "Introducing the cafe's name so late slows the payoff"; fix: *Name Bon Cafe earlier* |

| revision | scene | before | after |
|---|---|---|---|
| round 0 | 1 | Stop scrolling for the best Tashkent spot. Autumn chills require warm specialty coffee drinks today. | Step into the coziest autumn vibe with this rich spiced pumpkin latte. |
| round 0 | 3 | This hidden bakery serves freshly baked pastries that melt instantly in your happy mouth. | This hidden bakery serves freshly baked pastries that melt in your mouth completely. |
| round 0 | 5 | Save this exact address before your upcoming trip to explore stunning streets this weekend. | **Visit Bon Cafe on Mirobod street** before your upcoming trip to explore stunning spots. |
| round 1 | 1 | Step into the coziest autumn vibe with this rich spiced pumpkin latte. | Is this the coziest autumn coffee spot in Tashkent? |

What the producer remade: `round 1: voices 1, 3, 5`, `round 2: voices 1`. Every flagged problem was
"narration", so only voices were remade; the images and the hero clip were up to date (`approve_hero`:
"no new clip needed" both times). The replaced voices and each replaced final.mp4 are in `history/round_1/`
and `history/round_2/`.

**My reading.** The loop works mechanically: weak scenes went back, only they changed, the editor
re-rendered, the critic looked at the regenerated scenes first (round 1's frames include scene 5), and the
run stopped at the round limit. It did **not** converge: 5.5 -> 6.2 -> 6.0, a new set of complaints every
round, because the critic is blind across rounds (D9) and a single sample is noisy. Three failure modes worth
the lesson:
1. **The optimizer hallucinated** (E6): asked to "name the specific cafe", the reviser invented "Bon Cafe on
   Mirobod street". Fixed in the reviser prompt for later runs.
2. **The evaluator can't see a hallucination**: round 2 accepted the invented name as a fact and asked to
   name it *earlier*. A critic that judges persuasiveness rewards confident specifics, true or not.
3. **The critic wrote the fix itself**: round 1's suggested hook was used word for word by the reviser.
   Fine here, but the "writer" of that line is now the judge.
Still useful: rounds 0-1 removed a cliché opening and the "happy mouth" line, both real improvements.

### Scene 1 as a frame, and the vision path on a real picture

The critic's frames are kept in `critic/round_N/scene_NN.jpg` (540x960). In mock mode a frame is a coloured
card with the director's prompt printed on it, the burned-in subtitle, and (scene 1) the mock hero clip's
yellow "MOCK AI CLIP" label and colour drift. The critic reads the card's text: its `cannot_judge` said
"Actual visual quality, motion rendering, audio delivery, and pacing cannot be judged from text placeholder cards."

**Proof that it reads pixels, not just card text** (`vision_probe.py`, scratch, 1 call): a copy of run 1
with scene 4's image replaced by a real photograph already on this machine (Windows' lock-screen picture
`C:\Windows\Web\Screen\img102.jpg`, a lake at sunset with snowy hills; not copied into the repo), re-rendered,
one frame of scene 4 sent with `mock=False`. Scene 4's narration and prompt are about reading in a cozy
armchair by a rainy window. The critic:
```
"weak_scenes": [{"scene": 4, "problem": "visual",
  "reason": "The rendered visual shows an outdoor sunset over water instead of an indoor cafe armchair.",
  "fix": "Regenerate the frame to display a cozy cafe interior corner with an armchair and rainy window."}]
gemini-3.8-flash | 1908 in + 160 out | 590.6 s
```
It described the picture correctly and flagged the mismatch. One 540x960 frame = ~1,100 prompt tokens
(1,908 with the ~800-token text part).

**What can't be judged until real images exist:** image quality, consistency of faces/places across scenes,
artefacts (hands, text in images), whether the style words produce one look, the hero clip's motion, and
how the subtitles read over a busy picture. In mock mode the critic judges narration, the prompt-narration
fit and length; that is most of what it flagged in run 1 anyway.

---

## 4. Runs 2 and 3, and the producer over MCP

**Run 2** (`runs/2026-10-23_235059`, "Samarqand Registoni kechqurun", uz, `--research`) took five
invocations by design: crash before the director, resume into the approval pause, resume with `--approve
no` and a real kill in the producer, resume, then (after E11 stopped its first critic call) the last resume:
```
critic          37.0 s |   5432 in  1294 out | round 0: score 7.8, weak [6] (frames of scenes [1, 3, 4, 6])
done in 38 s | ledger lines for this run: 6 (1 new) | 12,693 tokens, $0.017798 list price | trace tr-4214cdc88ed7...
stop: passed (7.8 >= 7) | scores by round: [7.8]
'Registon kechasi' (uz)
scene  motion      planned   voice  on screen  words
1      zoom_in        3.2s    4.8s       4.2s      9          <- the refused hero scene: the director's zoom_in, no clip
...
total                18.0s   28.3s      25.2s     50
runs\2026-10-23_235059\final.mp4  1080x1920, 30 fps, 25.2 s, 3.5 MB, h264/yuv420p + aac
```
"1294 out" includes 1,076 thinking tokens (E13's fix was in by then). The critic passed it at 7.8 while
noting it was "slightly short": 50 words, a 25.2-second video, under the Uzbek target (55-60). The director's
plan (director.json) after E5's prompt fix: it now rewrites shots (wide / medium / extreme close-up / low
angle), changes 4 of 6 camera moves and sets one style, but keeps the writer's subjects; modest value.

**Run 3** (`runs/2026-10-24_001351`, "how to save money as a university student", en, `--research --mcp
--approve yes`), one invocation:
```
run 2026-10-24_001351 | ... | prompt v2 | max 2 revisions, pass at 7 | images: mock placeholders | tools over MCP
researcher       4.2 s |   1537 in   310 out | notes: 2 hook patterns, 2 angles
writer           6.4 s |   1439 in  1579 out | 3 options, picked 3: 'Your student ID is worth hundreds.'
director         5.0 s |    642 in   500 out | hero = scene 1: steam curls up and light glints off the surface of the student ID as the hand shifts slightly
?? Scene 1 (...) as a 5-second AI clip (...) costs ~$0.30 with real media; MOCK_MEDIA=1 makes a free placeholder clip now. Continue?
   answered by --approve: yes
approve_hero     0.0 s |                     | hero clip approved
producer        48.7 s |                     | round 0: images 1, 2, 3, 4, 5, 6; voices 1, 2, 3, 4, 5, 6; clip 1
editor          25.6 s |                     | runs\2026-10-24_001351\final.mp4
critic           9.2 s |   5444 in   123 out | round 0: score 8, weak none (frames of scenes [1, 3, 4, 6])
done in 100 s | ledger lines for this run: 6 (6 new) | 11,589 tokens, $0.011605 list price | trace tr-d781fccb3b60ac3bddba1aa92d9df8b9
stop: passed (8 >= 7) | scores by round: [8.0]
...  total 15.0s planned, 28.9s voice, 29.2s on screen, 74 words; final.mp4 1080x1920, 30 fps, 29.2 s, 4.9 MB
```
The hero motion "steam curls up ... off the surface of the student ID" is the director copying the example
in its own prompt ("steam curls up as a hand lifts the cup"): the example is gone from the prompt now (E14).
The same critic model that took 6-8 minutes per call in run 1's hour answered in 9 s here: queueing on the
free tier, not the model.

**The producer over MCP.** In run 3 every tool call went `producer -> McpToolbox -> stdio -> mcp_server.py
-> tools.py`: 13 calls (6 images, 6 voices, `animate_image`) in one server session; the producer node took
48.7 s (edge-tts dominates). The same plan re-made in a fresh folder by a scratch script with no LLM:
```
pass 1 over MCP: images 1, 2, 3, 4, 5, 6; voices 1, 2, 3, 4, 5, 6; clip 1 | 14.5 s
pass 2 over MCP: nothing (all up to date) | 2.0 s          <- just the server start: the manifest skipped everything
```
`animate_image` is published by the MCP server automatically (`registry_tool`), with a `run_id`, and its
description says it costs ~$0.30 and to ask the user first; the Phase 3 agent producer never sees it
(`tools.schemas()` still lists 3 tools: its own "clip" group).

---

## 5. Crash and resume (6.5): three real crashes, nothing paid twice

The checkpointer: `SqliteSaver` on `checkpoints/graph.sqlite` (gitignored; SQLite in WAL mode, so the
folder also holds `graph.sqlite-wal` and `-shm`), `thread_id = run_id`, `durability="sync"`. Cost of one
checkpoint, measured on run 1's final state (3 options, plan, 3 critic reports): **23.1 kB serialised,
0.7 ms per synchronous write**. Run 1 left 18 checkpoints (313 kB in all).

### Crash 1, not planned: Gemini answered 503 in run 1's third review

```
critic ... openai.InternalServerError: Error code: 503 - [{'error': {'code': 503, 'message': 'This model is
currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.',
'status': 'UNAVAILABLE'}}]
During task with name 'critic' and id '99be8ef4-...'
```
The OpenAI SDK had already retried twice (its default `max_retries=2`, short backoff) and `llm.paced()`
only handled 429s. The node failed, the process exited with code 1, and the checkpoint stayed at
"before critic, round 2". Fix (E9): `paced()` also waits out a 5xx (30 s, 60 s, 90 s).
Run 1's ledger lines before the resume (the `run_id` tag makes this one filter):
```
18:30:25 rag          gemini-embedding-2        17 in     0 out  $0.000003
18:30:27 researcher   gemini-3.5-flash-lite   1477 in   349 out  $0.001316
18:30:38 writer       gemini-3.5-flash-lite    793 in  2019 out  $0.005285
18:30:46 writer-pick  gemini-3.5-flash-lite    693 in   164 out  $0.000618
18:30:48 director     gemini-3.5-flash-lite    685 in   491 out  $0.001433
18:38:37 critic       gemini-3.8-flash        5322 in   288 out  $0.005071
18:38:43 revise       gemini-3.5-flash-lite    755 in   477 out  $0.001419
18:47:24 critic       gemini-3.8-flash        5321 in   186 out  $0.004688
18:47:26 revise       gemini-3.5-flash-lite    683 in   394 out  $0.001190
```
`make --resume 2026-10-23_233023` printed `resuming ... before critic` and made exactly one new call:
```
18:30:25 rag          gemini-embedding-2        17 in     0 out  $0.000003
18:30:27 researcher   gemini-3.5-flash-lite   1477 in   349 out  $0.001316
18:30:38 writer       gemini-3.5-flash-lite    793 in  2019 out  $0.005285
18:30:46 writer-pick  gemini-3.5-flash-lite    693 in   164 out  $0.000618
18:30:48 director     gemini-3.5-flash-lite    685 in   491 out  $0.001433
18:38:37 critic       gemini-3.8-flash        5322 in   288 out  $0.005071
18:38:43 revise       gemini-3.5-flash-lite    755 in   477 out  $0.001419
18:47:24 critic       gemini-3.8-flash        5321 in   186 out  $0.004688
18:47:26 revise       gemini-3.5-flash-lite    683 in   394 out  $0.001190
18:58:46 critic       gemini-3.8-flash        5318 in   213 out  $0.004787   <- new (the resume)
```

### Crash 2, injected: `--crash-before director` (a hard `os._exit(70)`, run 2)

```
$ make "Samarqand Registoni kechqurun" --lang uz --graph --research --crash-before director
run 2026-10-23_235059 | 'Samarqand Registoni kechqurun' (uz) | prompt v2 | max 2 revisions, pass at 7 | images: mock placeholders
researcher       3.7 s |   1583 in   317 out | notes: 2 hook patterns, 2 angles
writer           8.3 s |   1616 in  2203 out | 3 options, picked 1: 'Kechki Registonga men bilan birga yur.'
!! SHORTSMITH_CRASH_BEFORE=director: the process dies now, before 'director' starts        (exit code 70)
```
Ledger lines of run 2 at that moment:
```
18:51:02 rag          gemini-embedding-2     17 in      0 out  $0.000003
18:51:03 researcher   gemini-3.5-flash-lite  1583 in  317 out  $0.001267
18:51:10 writer       gemini-3.5-flash-lite   789 in 2040 out  $0.005337
18:51:12 writer-pick  gemini-3.5-flash-lite   827 in  163 out  $0.000656
```
`os._exit` skips every `finally` and every flush: the only things that survive are what was already on
disk, i.e. the checkpoint written (sync) after the writer, and those four ledger lines.

### Resume without an answer: the director runs, then the run pauses at the approval question

The first try exposed a bug (E10): on Windows `sys.stdin.isatty()` is True for the NUL device, so the CLI
called `input()` and died with `EOFError`. Nothing was lost: the director had run and the interrupt had
been checkpointed. After the fix:
```
$ make --resume 2026-10-23_235059
resuming 2026-10-23_235059 before approve_hero ('Samarqand Registoni kechqurun')
?? Scene 1 ("Kechki Registonga men bilan birga yur, sehrli manzarani ko'rsataman.") as a 5-second AI clip
   (Golden architectural lights gently pulse and illuminate the grand madrasah facades as the camera slowly
   pushes forward into the square.) costs ~$0.30 with real media; MOCK_MEDIA=1 makes a free placeholder clip now. Continue?
   continue? [y/N]
   no one to ask: the run is saved and paused here.
paused in 0 s | ledger lines for this run: 5 (0 new) | 7,043 tokens, $0.008872 list price | trace tr-5e1685ae...
resume with:  uv run --env-file .env python -m shortsmith.make --resume 2026-10-23_235059 --approve yes|no
```
5 lines = the 4 above + the director's (754 in, 553 out), made once in the EOFError attempt; "0 new" here.

### Crash 3, real: `Stop-Process -Force` in the middle of the producer

`kill_producer.ps1` (scratch, not in the repo) started `make --resume 2026-10-23_235059 --approve no`,
watched `audio/`, and killed both python processes (the venv's `python.exe` launcher and the interpreter
it starts) as soon as 3 voice files existed:
```
approve_hero     0.0 s |                     | hero clip refused: Ken Burns instead
killed python pid 16128 with 3 voice files made
killed python pid 19148 with 3 voice files made
```
On disk: images 1-3, voices 1-2 with their timings, and `audio/scene_03.mp3` **with 0 bytes and no
timings file**: the kill landed while edge-tts was writing it. media.json listed image+voice for 1-2 and only
the image for 3 (an entry is written after a tool returns, never before). The checkpoint said
`next: ('producer',)`, no error (a hard kill leaves none). Resume:
```
$ make --resume 2026-10-23_235059
resuming 2026-10-23_235059 before producer ('Samarqand Registoni kechqurun')
producer         9.5 s |                     | round 0: images 4, 5, 6; voices 3, 4, 5, 6
editor          17.9 s |                     | runs\2026-10-23_235059\final.mp4
```
Voice 3 was made again because the manifest never recorded it; the 0-byte file went to
`history/round_0/audio/scene_03.mp3` (ffprobe: "Failed to find two consecutive MPEG audio frames"), the new
one is 4.68 s. A file-exists check would have kept the empty file and broken the edit. The approval answer
("no") wasn't asked again: `approve_hero` had finished and was checkpointed before the producer started.

What a checkpoint can't save: a node killed half-way runs again from its start. Here that cost nothing
(the producer has no LLM calls and skips finished files), which is the reason paid work sits in small
nodes and the producer is idempotent. An LLM node killed mid-call loses that call: its answer never
reached the ledger, but the provider may have counted it.

---

## 6. The approval interrupt (6.6)

`approve_hero` runs before every producer pass; it pauses only when this plan would make a new paid clip
(`producer.clip_needed()`: approved isn't False, the hero scene is still `ai_clip`, and the clip is missing
or was made from another image/motion). What the pause looks like, in order:

1. The node calls `interrupt(HeroQuestion(question, scene, motion, cost_usd, mock).model_dump())`. LangGraph
   saves the state and the pending interrupt; `stream()` ends; `get_state().next == ('approve_hero',)` and
   `.interrupts[0].value` is the question.
2. `drive()` asks the `Approver`: `--approve yes|no` answers at once; a terminal gets `continue? [y/N]`;
   otherwise (or on EOF, E10) the run pauses and prints the resume command. The process can exit here.
3. `Command(resume=answer)`: LangGraph re-runs `approve_hero` from its first line; this time `interrupt()`
   returns the answer. Yes -> `hero.approved = True`, the producer calls `animate_image`. No ->
   `hero.approved = False` (never asked again in this run) and the hero scene's motion becomes the
   director's Ken Burns move (`hero.fallback`): the video still finishes.

Seen live: "yes" in runs 1 and 3 (a mock clip `clips_ai/scene_01.mp4`, 540x960, 5 s, colour drift + sway +
"MOCK AI CLIP" label; the editor fits it to the scene: `clip_cmd` loops it under a longer voice, run 1's
scene 1 shows `ai_clip` in the summary); the pause and "no" in run 2 (scene 1 = `zoom_in`, no clip). In run 1
the two later rounds printed "no new clip needed": the revisions changed narration only, so the approved
clip was still up to date. Tests: yes/no/pause-then-resume, a revision of the hero image asking again, and
the CLI's pause message (`test_the_cli_runs_pauses_and_resumes_a_graph_run`). The real fal path
(`media/animate.py: _fal`: upload, `fal_client.subscribe`, ledger line, then download) was never run; a test
fakes fal and checks the order (price recorded before the download) and the arguments. The model id
(`fal-ai/kling-video/v2.1/standard/image-to-video`) and the $0.30 estimate must be checked on fal.ai first.

---

## 7. MLflow tracing (6.7)

`tracing.setup()` (called by `make --graph` unless `--no-trace`, and by the evaluation) sets
`MLFLOW_DISABLE_TELEMETRY=true`, `DO_NOT_TRACK=true` and `MLFLOW_DISABLE_AGENT_HINT=1` before importing
mlflow, points it at `sqlite:///mlflow.db` (repo root, gitignored; trace attachments in `mlruns/`, also
gitignored), sets the experiment `shortsmith`, and turns on `mlflow.openai.autolog()`. Every MLflow call is
wrapped: run 1 started with MLflow broken (E1) and simply ran untraced, with one warning line.

**Does autolog see Gemini?** Yes. llm.py uses the OpenAI SDK with Gemini's base URL, and autolog patches
the SDK: each call is a `Completions [CHAT_MODEL]` span with the exact messages, the raw answer, the
response_format schema, `mlflow.chat.tokenUsage` and even `mlflow.llm.cost` (MLflow ships a price table:
for run 1's last critic call it computed $0.004787, the same as our ledger). It also caught the
researcher's embedding request (`Embeddings [EMBEDDING]`). MLflow 3.17 stores each base64 frame of the
critic's request as a trace attachment (`mlflow-attachment://...`, files under
`mlruns/<experiment>/traces/<id>/artifacts`), so the UI shows the frames inline and the span text stays
small (a span processor I wrote to shrink them found nothing to shrink and was removed).

### One full run, read span by span: run 3, trace `tr-d781fccb3b60ac3bddba1aa92d9df8b9` (15 spans, 99.3 s)

```
run 2026-10-24_001351 [AGENT] 99.32 s OK        inputs {"topic": "how to save money as a university student", "resume": false}
  researcher [AGENT] 4.19 s   calls=1 in=1537 out=310      reads {"topic", "language": "en", "brand": null}
    Embeddings [EMBEDDING] 1.07 s                          "task: search result | query: how to save money ..."
    Completions [CHAT_MODEL] 1.81 s  tokens 1537/310       the researcher's system prompt + trends + 6 patterns
  writer [AGENT] 6.44 s       calls=2 in=1439 out=1579     reads {"prompt_version": "v2", "notes_tokens": 251}
    Completions 4.52 s  tokens 795/1408                    3 options (ScriptOptions schema)
    Completions 1.55 s  tokens 644/171                     the pick: scores 8/9/9, 7/8/8, ... best 3
  director [AGENT] 4.98 s     calls=1 in=642 out=500       reads {"chosen": {"title": "Campus ID Goldmine", ... "words": 74}, "visual_style": ""}
    Completions 2.08 s  tokens 642/500
  approve_hero [CHAIN] 0.00 s                              (the pause: attributes {"interrupted": true}, outputs null)
  approve_hero [CHAIN] 0.00 s                              (the resumed run: outputs hero.approved = true)
  producer [CHAIN] 48.70 s                                 outputs {"made": ["round 0: images 1-6; voices 1-6; clip 1"]}
  editor [CHAIN] 25.57 s                                   outputs {"final_path": "runs\\2026-10-24_001351\\final.mp4"}
  critic [AGENT] 9.23 s       calls=1 in=5444 out=123      outputs {"critic_rounds": [{"round": 0, "score": 8.0, "frames": [1, 3, 4, 6], "weak": []}]}
    Completions [CHAT_MODEL] 7.92 s  tokens 5444/123 (total 6184)   gemini-3.6-flash; 4 frames as attachments
```
What it shows at a glance: where the time went (producer 49% and editor 26% of this run: edge-tts and
ffmpeg, not the LLMs, once the free tier isn't queueing), where the tokens went (the critic's 5,444-token
prompt is 4 images), what each node was given (the director got the chosen script and an empty style, no
notes), the interrupt as two `approve_hero` spans, and the exact prompt behind any output. Reading it this
way found three bugs, all fixed: the hidden thinking tokens (`total_tokens` 6,184 vs 5,444 + 123, E13), the
style words appended twice ("photorealistic, 8k resolution, photorealistic, 8k resolution", E15), and my
class docstrings sent to the model inside the response_format schema (E16).

**Traces of crashed invocations** (8 traces in `shortsmith`, one per invocation, tagged `run_id` and
`resumed`): MLflow 3.17 exports spans as they finish, so a hard-killed process leaves an `IN_PROGRESS`
trace holding only the spans that finished before the kill, with no root span and no tags (run 2's
injected crash: just `researcher`; the `Stop-Process` kill: two `approve_hero` spans, no producer). The
`EOFError` invocation is a complete trace with state `ERROR`. Tracing tells you what happened up to a crash;
it is not what makes a run resumable: the checkpoint is.

### The owner's steps to open it (checked in the built-in browser on 2026-10-24)

Terminal (PowerShell), repo root:
```powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
$env:MLFLOW_DISABLE_TELEMETRY = "true"
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then open http://127.0.0.1:5000 and:
1. **Experiments -> shortsmith -> Traces** (left menu): one row per graph invocation. Click the row whose
   input is `{"topic": "how to save money as a university student", "resume": false}`.
2. Left pane: the span tree above. Click a node span (e.g. **writer**) for its Inputs/Outputs (the
   summaries) and **Attributes** (calls, prompt_tokens, completion_tokens, cost_usd, seconds).
3. Click a **Completions** span under it: the exact system and user messages and the model's raw JSON
   answer; under **critic -> Completions**, "See more" in the user message shows the 4 frames the critic saw.
4. **Experiments -> shortsmith-prompt-eval -> Runs** (the 6.8 evaluation): 10 runs named `v1 | topic` /
   `v2 | topic` under a parent run. Add the columns `prompt_version`, `critic_score`, `options_in_range`,
   `options_hook_copied`, or tick runs and press **Compare**.
Ignore the UI's "MLflow Assistant" panel and the "Detect issues" button: they want an LLM of their own.
Stop the server with Ctrl+C. (The server prints "MLflow job backend does not support Windows": harmless.)

### Trace vs log, the same run (Q3)

The log: `runs/ledger.jsonl`, the 6 lines with `"run_id": "2026-10-24_001351"`:
```
19:13:54 rag          gemini-embedding-2      15 in      0 out  $0.000003  (tokens estimated)
19:13:56 researcher   gemini-3.5-flash-lite 1537 in    310 out  $0.001236
19:14:01 writer       gemini-3.5-flash-lite  795 in   1408 out  $0.003759
19:14:02 writer-pick  gemini-3.5-flash-lite  644 in    171 out  $0.000621
19:14:07 director     gemini-3.5-flash-lite  642 in    500 out  $0.001443
19:15:31 critic       gemini-3.6-flash      5444 in    123 out  $0.004544  (reasoning 0: written before E13's fix; 617 really)
```
plus the console lines (one per node). The log answers "what was paid, by which step, when": flat,
append-only, written before a response is even parsed, survives every crash (all of run 2's crashes left
complete ledger lines). The trace answers "how did it unfold": the tree (which call belongs to which node),
durations including the 48.7 s of the producer that the ledger can't see (no paid call), each node's inputs
and outputs, and the full prompt and answer of every call. Neither replaces the other: the ledger is the
budget's source of truth (once E13 was fixed), the trace is where E13, E15 and E16 were spotted.

---

## 8. Evaluation (6.8): writer prompt v1 vs v2

`SHORTSMITH_CRITIC_MODEL=gemini-3.6-flash uv run --env-file .env python -m shortsmith.agents.prompt_eval` ->
`runs/experiments/phase6_prompts_2026-10-24_000657/` (results.json with every option, script and review;
report.txt) and MLflow experiment **shortsmith-prompt-eval**: a parent run (`v1_mean_critic_score`,
`v2_mean_critic_score`) and 10 nested runs, params `topic / language / prompt_version`, metrics
`critic_score, weak_scenes, chosen_words, chosen_in_range, chosen_hook_copied, options_valid,
options_rejected, options_in_range, options_hook_copied, writer_*_tokens, writer_cost_usd, critic_prompt_tokens,
seconds`. 19:06-19:13 UTC.

**Setup.** 5 topics (3 English, 2 Uzbek). Per topic one research step, shared by both versions; then the
Writer exactly as the graph runs it (3 options, checks, pick) with v1 and with v2; then the Critic in
script mode on the chosen script. The judge is gemini-3.6-flash (E11: 3.8 Flash's 20 free requests for the
day were gone), a full Flash model, stronger than the writer (3.5-flash-lite), never told the version.
Which version went first alternated per topic. One sample per topic and version, default temperature.

**What v2 changes** (writer.PROMPTS): exact words per scene with "outside the range is thrown away"; "hook
patterns in the notes are ideas, not sentences: never reuse an example hook"; "the pictures SHOW, the
narrator TELLS: never describe what the picture shows"; and in the notes: hook patterns WITHOUT a ready-made
sentence, angles split into what the narrator SAYS and what the picture SHOWS.

| topic | ver | critic score | weak | words (target) | in range | hook copied | options in range | options copied | hook |
|---|---|---:|---:|---:|---|---|---:|---:|---|
| coffee shop in Tashkent, cozy autumn vibe (en) | v1 | 7.0 | 1 | 67 (78-84) | no | **yes** | 0/2 (+1 rejected) | 2/2 | "POV: you found Tashkent's coziest autumn cafe." |
| | v2 | **8.5** | 0 | 82 | yes | no | 3/3 | 0/3 | "Your secret autumn escape is here." |
| how to save money as a university student (en) | v1 | **8.0** | 0 | 72 | no | **yes** | 0/3 | 3/3 | "Save this before your next semester starts." |
| | v2 | 7.0 | 1 | 67 | no | no | 0/3 | 0/3 | "Save this for your campus wallet." |
| a day trip from Tashkent to the Chimgan mountains (en) | v1 | **8.0** | 0 | 74 | no | no | 0/3 | 1/3 | "Escape Tashkent heat in one hour." |
| | v2 | 7.5 | 1 | 79 | yes | no | 1/3 | 0/3 | "Escape the city heat right now." |
| Samarqand Registoni kechqurun (uz) | v1 | 6.5 | 2 | 47 (55-60) | no | no | 0/3 | 2/3 | "Kechasi Registonga kirish mumkinmi?" |
| | v2 | **7.5** | 2 | 60 | yes | no | 3/3 | 0/3 | "Tungi Samarqand nafasini his qiling." |
| uyda palov tayyorlash sirlari (uz) | v1 | **7.5** | 1 | 57 | yes | **yes** | 2/3 | 3/3 | "Oshning siri go'shtda deb o'ylaysizmi?" |
| | v2 | 6.5 | 1 | 51 | no | no | 0/2 (+1 rejected) | 0/2 | "Bibi aytgan qoida tamom boshqacha." |
| **all 5** | v1 | **7.40** | 4 | | **1/5** | **3/5** | **2/14** (+1 rej.) | **11/14** | |
| | v2 | **7.40** | 5 | | **3/5** | **0/5** | **7/14** (+1 rej.) | **0/14** | |

Writer cost: v1 7,119 in + 9,175 out tokens ($0.0251 list price), v2 7,470 + 9,699 ($0.0265): v2's longer
rules cost ~5% more. 35 LLM calls (5 research, 10 options, 10 picks, 10 critic), $0 billed.

**What the numbers say.**
- **On what v2 targeted, it worked, and these are measured by code, not by a judge:** copied hooks went from
  11 of 14 options (79%: the notes' suggested sentence or a library example, word for word) to **0 of 14**;
  options inside the word target from 2/14 to **7/14**; chosen scripts in range from 1/5 to 3/5. Narration
  that describes the picture: the critic flagged it once, for v1 (coffee shop, scene 3: "merely describes
  what the visual prompt shows"); never for v2.
- **On overall quality, no difference: 7.40 vs 7.40.** v2 won 2 topics, v1 won 3. With one sample per topic
  and score differences of 0.5-1.5, a 2-3 split is what "no difference" looks like; Phase 5 measured the
  same judge setup moving a script by up to 2 points between runs.
- Length is still not solved: 7 of 14 v2 options are outside the range (mostly a few words short), and v2's
  Uzbek plov script was 51 words (target 55-60).

**My own reading (all 10 scripts, the Uzbek ones too).**
- *Coffee shop: v2 clearly better, agree.* v1 is the Phase 5 failure again: "Golden autumn light streams
  through large windows onto cozy velvet armchairs" is a caption, not narration, and the hook is the
  notes' sentence. v2 talks to the viewer ("Share this video with a friend and plan your coffee visit") and
  hits 82 words. Still a little sensory ("Feel the gentle heat radiating from your mug").
- *Saving money: v1 better, agree.* v1 is concrete and conversational ("Always ask cashiers if they have a
  student deal available today"); v2 is stiff ("valid university registration documents", "flash your
  plastic"). v1's hook is the researcher's suggested sentence; the content is better anyway.
- *Chimgan: about equal.* v2 pads to reach the word count ("...bloom across these slopes during the pleasant
  summer afternoon hours"), the side effect of "count your words". v1 tells a first-person story ("We arrived
  early to catch the cable car") nobody lived. v2 lost half a point to a critic bug (below).
- *Registan: v2 better, agree.* v1 has 47 words (a ~23-second video) and a hook question ("Can you enter
  Registan at night?") it never answers. v2 has the right length but is bookish ("Me'moriy yechimlar har bir
  kishini o'ziga rom etishga qodirdir"): neither sounds like spoken Uzbek.
- *Plov: v1 better, agree.* v2 says "use suzma instead of ordinary rice" (Oddiy guruch o'rniga suzma), which
  is nonsense (suzma is strained yogurt); the critic caught it. v1 has concrete steps (soak the rice, lower
  the heat for the zirvak) but its hook is copied from the notes.
- **Critic bug found by reading (E12):** in 2 of 10 reviews (v1 Registan, v2 Chimgan) the critic flagged
  "scene 1 repeats the hook verbatim". Our rule is that scene 1 *starts with* the hook; the critic's text
  showed a separate `Hook:` line, so it looked spoken twice. Fixed (the line now says the hook is the opening
  of scene 1, spoken once); one false flag per version, so it doesn't change the comparison.

**How do we know v2 is better? (Q4)** We know it is better at what it was written for, because those are
objective and measured on every option: 0 copied hooks instead of 79%, 3.5x more options inside the word
target. We do **not** know it writes better videos: the blind, stronger judge scored both 7.40, and my
reading agrees it's 2-2 with one tie. What would make a claim stronger: more topics (20+) and 3 samples per
topic at temperature 1 (variance), a second judge model (or a human panel), and a paired test on the score
differences. Decision for now: v2 stays the default, because copied hooks (two videos opening with the same
line) and wrong lengths (a 23-second video) are defects regardless of taste; v3 should go after v1's strength,
concrete tips and steps, and against padding.

---

## 9. Context engineering (6.9): trimmed vs whole-state requests, per node

What each LLM node reads (the trimmed design, from the start of this phase, visible in every node span's
inputs):

| node | reads | never sees |
|---|---|---|
| researcher | topic, language, brand; trends, 6 retrieved patterns, 3 brand sections (Phase 5) | |
| writer (options) | versioned rules + topic + the notes rendered for the writer (~245-300 tokens) | trend data, scores, URLs, other runs |
| writer (pick) | each option's spoken words + code facts (words, copied hook) | image prompts, notes |
| director | the chosen script (narration + the writer's picture idea) + brand style words | notes, trends, the other options |
| critic | the final script (narration + image prompts), code facts, 4 frames | notes, trends, options, earlier reports |
| revise | the script, word targets, the flagged scenes with reasons and fixes | notes, trends, frames, options |
| producer, editor | the plan (no LLM) | research notes (the Editor never sees them, PLAN §2) |

The "before" is the common shortcut: give every agent the whole state. `agents/context_eval.py` rebuilds
each node's exact request from run 1's checkpoints (the same builder functions, so the trimmed request is
byte for byte what the node sent: a test checks this), appends the whole RunState as JSON plus research.json
(what a pass-everything pipeline would carry), and sends that once per node (ledger steps `ctx-full-*`).
"trimmed" is the provider's prompt_tokens from the run's own ledger line; "whole state" is the provider's
count for the ctx-full request; the tiktoken estimates are in brackets.

| node | trimmed (real) | whole state (real) | x | what the extra ~4-7k tokens are |
|---|---:|---:|---:|---|
| writer (options) | 793 (736 est.) | 4,707 (4,109 est.) | **5.9x** | research.json (10 trend videos with ids, URLs, counts; 6 retrieved patterns with scores) + state settings |
| writer-pick | 693 (651) | 6,704 (6,092) | **9.7x** | + the 3 options in full (image prompts too) |
| director | 685 (700) | 6,741 (6,142) | **9.8x** | the same: options, notes, trends it can't use |
| revise | 755 (810) | 7,957 (7,297) | **10.5x** | + the plan, the first critic report, the chosen script again |
| critic | 5,322 (911 est. text) | not sent: 3.6 Flash's daily quota was gone (E11); ~12,100 projected (7,085 est. text + the same images) | ~2.3x | the images (~4,400 tokens) are the same in both; the text part grows ~ninefold |

Reading it:
- Trimming cuts the text prompts of the four flash-lite nodes by **6-10x**. In this run the four nodes would
  have read **26,109** prompt tokens with the whole state instead of **2,926**. On the free tier that is
  "only" latency and quota; at the paid list price ($0.30/1M for flash-lite) it's $0.0078 vs $0.0009 per run
  for those nodes, i.e. it scales with every run and every round (revise and critic run up to 2-3 times).
- The whole state grows during a run (options, reports, revisions), so late nodes pay most: revise 10.5x.
  Checkpointing doesn't care (0.7 ms per write); the prompt does.
- The critic is dominated by its images: trimming its text saves proportionally less. The lever there is
  the number of frames and their resolution (open item: Gemini's `media_resolution` for placeholder cards).
- Beyond tokens: a node that sees research notes can act on them. The Director given the notes is exactly
  how Phase 5's coffee-shop loss happened in the other direction (the notes' visual angles leaked into the
  narration). Not measured as quality here (one call per node, outputs not compared): the claim is "fewer
  tokens and fewer distractions", and only the first half has numbers.
- tiktoken (o200k) is within ±7% of Gemini's count on the trimmed prompts (both directions) and 8-13% under
  on the whole-state ones: JSON (quotes, braces, ids, URLs) costs Gemini relatively more tokens.

---

## 10. Errors hit, how they were found, how they were fixed

**E1. `ImportError: DLL load failed while importing _uuid_utils: An Application Control policy has blocked
this file.`** The very first `import langgraph` (6.1). The chain: `langgraph.graph` -> `langchain_core.runnables`
-> `langchain_core.utils.uuid` -> `uuid_utils` 0.18.0, a native (Rust) wheel. "Application Control policy"
is Windows 11's Smart App Control: it blocks unsigned native binaries that have no reputation yet, and a wheel
published days ago has none (langchain-core 1.7.0 itself was uploaded on 2026-10-22). I tested every new
native module one by one: `ormsgpack`, `xxhash`, `zstandard`, `sqlite_vec`, `pyarrow`, `scipy` loaded;
`uuid_utils` 0.18.0, `pandas` 3.1.0 (`pandas_datetime`) and scikit-learn (`_cyutility`) were blocked. Later,
run 1's tracing found a fourth: SQLAlchemy 2.1.6 (`_immutabledict_cy`), imported by MLflow's tracking store.
Fix, without touching any security setting (that's the owner's machine policy, and we never change it):
`[tool.uv] constraint-dependencies` pins versions whose DLLs load: `uuid-utils==0.12.0`, `pandas==2.3.3`,
`sqlalchemy<2.1` (SQLAlchemy 2.0 also falls back to pure Python when a compiled module fails to import;
2.1 doesn't). scikit-learn is never imported by our code paths, so it stays as it is. Constraints, not
dependencies: they pin a version only if something else needs the package.
Run 1 ran with MLflow broken this way: `tracing off: ImportError: ... _immutabledict_cy ...` on stderr and
the run carried on untraced, the "tracing must never break a run" rule tested for real before any test did.

**E2. `ImportError: Install grandalf to draw graphs`.** `get_graph().draw_ascii()` needs an extra package.
Fix: `draw_mermaid()` (Mermaid text, no dependency; paste into mermaid.live).

**E3. `SyntaxWarning: invalid escape sequence '\-'`** from the ASCII diagram in toy_graph.py's docstring
(`\--`). Fix: a backtick instead of the backslash.

**E4 (test bugs, found by the tests).** (a) I asserted 7 LLM calls for the loop run; it's 6 (writer 2,
director 1, critic 2, revise 1). (b) `pytest.approx` inside a Pydantic constructor: Pydantic rejects it as
not a float. (c) A 3-scene test script: `Script` requires 4-8 scenes. (d) The exercise's recursion-limit
test passed against the empty exercise: `NotImplementedError` is a subclass of `RuntimeError`; the test now
asserts the error is not a NotImplementedError. (e) A fake `animate_image` that didn't create `clips_ai/`.
(f) `context_eval.requests()` tried to build a RunState from the first checkpoint, whose values are empty
(it's the one before `__start__`); only snapshots whose next node is a graph node are used now.

**E5. The Director handed the writer's image prompts back unchanged (run 1).** Found by looking at a
critic frame: the placeholder card showed the writer's wording. script.json vs options.json: all 6 prompts
and camera moves identical. The model treated "writer's picture idea" as final. Fixes: the prompt says the
idea is a rough note to rewrite as a shot (vary close-ups and wide shots), never to hand back unchanged;
`director.json` now saves the director's raw answer, so the next run shows whether it rewrote anything.

**E6. The Reviser invented a business (run 1).** The critic flagged scene 5 ("claims to provide an exact
address without naming the venue"; fix: "name the specific cafe"), and the reviser wrote "Visit Bon Cafe on
Mirobod street ...": a name and a street nobody gave it (a real chain name, which is worse). The critic was
right that the line made an empty promise; the fix it suggested asked for a fact the system doesn't have.
Fix: the reviser prompt now forbids inventing names, addresses, prices, dates or numbers and says to make
the line concrete another way or drop the promise. Lesson: in an evaluator-optimizer loop the evaluator's
suggestions are input to a generator, and they can push it into hallucination.

**E7. A revised hook cut mid-phrase (run 1).** Scene 1's new narration was one 12-word sentence; my
`_first_sentence(max_words=10)` made the hook "Step into the coziest autumn vibe with this rich spiced".
Fix: the reviser now returns the hook itself (`SceneFixes.hook`); code uses it only if scene 1's narration
really starts with it, else the whole first sentence (never truncated).

**E8. MLflow 3.17 prints an "agent hint" on import** (`Load the instrumenting-with-mlflow-tracing skill ...`
on stderr). Harmless, but noise in every CLI run: `tracing.setup()` sets `MLFLOW_DISABLE_AGENT_HINT=1`.

**E9. `openai.InternalServerError: Error code: 503 - ... 'This model is currently experiencing high demand.
Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'`** in run 1's third
critic call. The SDK's own retries (2, short backoff) weren't enough and `paced()` handled only 429s, so the
run crashed in the critic node (the checkpoint saved it: §5). Fix: `paced()` also waits out any 5xx, 30 s ×
attempt, up to 4 attempts; a test fakes two 503s and checks the waits are [30, 60].

**E10. `EOFError: EOF when reading a line`** on `make --resume` without `--approve`. The CLI asks only when
`sys.stdin.isatty()`, and on Windows that is True for the NUL device my shell gives a process. Found from
the traceback (the director had run, and the interrupt was saved: nothing lost). Fix: EOF means "nobody to
ask": the run pauses and prints the resume command; a CLI test runs pause -> resume with "no" -> "already finished".

**E11. `429 ... Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests,
limit: 20, model: gemini-3.8-flash ... quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`.** The big
one. At ~19:00 UTC every critic call (run 1's resume had just finished; the vision probe, run 2 and the
evaluation were all waiting on 3.8 Flash) started getting 429s. My warning printed only the first 300
characters, which stop right before the metric's name, so I couldn't tell a per-minute limit from a daily
one. One tiny probe call per model (a scratch script printing only the quota fields) answered it: flash-lite
fine, **3.8 Flash at its daily limit of 20 requests**, which Phase 5's 8 judge calls and this phase's
critic calls had used up (refused requests may count too). It resets at midnight Pacific (07:00 UTC), 12 hours
later. Fixes: (1) I stopped the stuck evaluation and run 2 (run 2 resumed later from its checkpoint; the
evaluation's first attempt had done 4 calls, wasted); (2) the critic model is configurable,
`SHORTSMITH_CRITIC_MODEL`, default still gemini-3.8-flash; today's runs used **gemini-3.6-flash** (a full
Flash model, stronger than the writer, its own 20 a day; price added to the ledger from Google's pricing
page: the same $0.75/$3.75 as 3.8 Flash; gemini-3.7-flash is not on the pricing page, so not used); (3)
`paced()` raises `DailyQuotaError` at once on a per-day 429 instead of retrying 4 times (a minute's wait
can't fix a daily quota), and the warning now shows the error text. By the end 3.6 Flash was out too: the
context measurement's critic step got one `DailyQuotaError` and was recorded as an estimate. Lesson: on the
free tier the judge's daily quota, not money, is the evaluation budget: ~20 critic calls a day per model.

**E12. The critic flagged "scene 1 repeats the hook verbatim" (2 of 10 evaluation reviews).** Found by
reading the reviews next to the scripts: our rule is that scene 1 *starts with* the hook (spoken once), but
`script_text()` showed it on its own `Hook:` line first. Fix: the line reads "Hook (the opening words of
scene 1, spoken once)". One false flag per version, so the v1/v2 comparison is unaffected.

**E13. The ledger didn't count thinking tokens.** Found in run 3's trace: the critic's raw usage was
`prompt_tokens 5444, completion_tokens 123, total_tokens 6184, completion_tokens_details null`. Gemini 3
Flash models think before answering and the compatible endpoint reports the thinking only inside
`total_tokens`; Google bills it as output. Every flash-lite call had total = prompt + completion (it doesn't
think by default), every Flash call didn't (3.8 Flash: +350 on a critic call). Fix in `_complete()`, the one
place calls are priced: `reasoning = total - prompt - completion` (0 for OpenAI, whose completion count
already includes reasoning), priced at the output rate, recorded as `ChatResult.reasoning_tokens` and a new
optional `LedgerEntry.reasoning_tokens`; `Usage.completion_tokens` includes it. Size of the gap before the
fix: across the 12 traced 3.6 Flash calls, **10,955 thinking tokens vs 2,093 visible output tokens**: ~9,900
unpriced tokens, ≈$0.037 at list price ($0 billed). Phase 5's 8 judge calls (3.8 Flash) were under-priced the
same way. The ledger is append-only, so old lines stay as written. MLflow's own `mlflow.llm.cost` makes the
same mistake ($0.004544 for that call: prompt + visible output only).

**E14. A prompt's example came back as output.** Run 3's director wrote the hero motion "steam curls up and
light glints off the surface of the student ID": the example in its own prompt was "steam curls up as a hand
lifts the cup". Fix: the prompt describes the shape of the answer ("the subject's movement, then the
camera's") without a concrete example.

**E15. Style words added twice.** The critic's request in run 3's trace showed prompts ending
"photorealistic, 8k resolution, photorealistic, 8k resolution, vertical 9:16 ...": the director had already
ended a prompt with part of its own style, and `with_style()` only skipped the style when the WHOLE string
was present. Fix: it appends only the comma-separated parts the prompt doesn't contain; a test covers it.

**E16. My docstrings were sent to the model.** Also in run 3's trace: the critic's `response_format` schema
contained the description "What the critic model writes. A strict JSON schema can't express dict[int, str]
...". Pydantic copies a model's docstring into its JSON schema. Fix: developer notes on LLM schema classes
(`CriticVerdict`, `ScriptOptions`) are comments now; a test asserts no `description` in those schemas. (Phase
5's `ResearchNotes` docstring is still sent, on purpose: it describes the notes to the researcher.)

---

## 11. Design decisions and why

**D1. LangGraph standalone, our own LLM wrapper.** Only `langgraph` and `langgraph-checkpoint-sqlite` are
imported; every model call still goes through `llm.py` (one ledger, one provider switch, our 429 handling).
LangGraph nodes are plain functions; they don't care what calls the model. langchain-core comes along as a
hard dependency of langgraph (its `RunnableConfig`, callbacks and graph drawing), and it is the package that
pulled in the blocked `uuid-utils` DLL (E1): "without LangChain" means without LangChain's model layer, not
without its code on disk.

**D2. A Pydantic `RunState`.** The same as every other data model in the project: types are checked when a
node's state is built, nested models (Script, CriticReport) round-trip through the checkpointer, and the
fields read like PLAN §5. Two surprises of a Pydantic state, both measured in 6.1: `invoke()` and
`get_state().values` return a plain dict (so `RunState.model_validate(values)` after a run), and fields
never written are absent from that dict (defaults are not channel values).

**D3. Reducers only where several writes must add up.** `critic_rounds`, `revisions`, `made`: append
(`operator.add`); `usage`: a custom per-node sum (`add_usage`). Everything else is replaced. Nodes return
deltas, never totals: no node reads `state.tokens_used`, adds its own and writes it back, the
read-modify-write pattern that loses updates as soon as two nodes run in the same step. `tokens_used` and
`cost_usd` (PLAN §5) became properties computed from `usage`.

**D4. Writer: 1 call for 3 options + 1 pick call (2 calls, not 4).** Options written side by side differ
on purpose (asked for different hook patterns and angles), for one prompt's price. Each option is validated
alone (`chat_schema` + `Script.model_validate` per option): one option breaking the word rule doesn't throw
away the other two, which `chat_json` would. The pick reads spoken words only (no visual prompts), plus
facts code measured (word count vs target, hook copied or not). The pick is skipped when only one option
survives. Self-pick by the same model is what PLAN asks for; the 6.8 judge is a different model.

**D5. Director as its own LLM step.** It owns the picture side: one style for all prompts, shots with
composition, a camera move per still, and the choice of the single paid hero scene with its motion prompt.
It reads the chosen script and the brand's style words only. Code applies the plan (narration untouched,
style appended where missing, hero motion = `ai_clip`, the director's own move kept as the fallback).

**D6. Producer = deterministic code, not the Phase 3 agent.** The steps are known (every scene needs an
image and a voice; maybe one clip): a workflow. An agent producer spent 3-4 LLM turns in Phase 3; on the
free tier each turn is 7-300 s, and its decisions add nothing. The producer works like `make`: media.json
remembers which prompt made each image and which words each voice says; it (re)makes only what is missing
or out of date, so (a) a revision regenerates only the changed scenes and only the changed part, (b) a
crash inside the producer resumes for free, (c) the old file is archived in `history/round_N/`. Tools are
reached through `call_tool(name, json)`, so `--mcp` sends the same calls through the MCP server.

**D7. The revision is its own node (`revise`) between Critic and Producer.** Evaluator and optimizer
stay separate (a critic that rewrites tends to grade its own fix), the producer stays LLM-free, and the
revision gets its own checkpoint (a crash in the producer doesn't pay for the revision again) and its own
span. The critic's `fix` goes into the prompt; code checks that the field really changed and the producer
only remakes changed media.

**D8. The loop's rules.** Revise when the last report has weak scenes AND score < `pass_score` (7);
at most `max_rounds` (2) revisions, i.e. at most 3 critic reports; a passing score ends the run even with a
nitpick; a low score with nothing flagged ends it too (nothing to fix). At most 3 weak scenes per round
(`CRITIC_MAX_WEAK`): flagging more than half a video is a rewrite, not a fix.

**D9. Critic: 4 frames, chosen by rule.** Just-regenerated scenes first, then an even spread that always
includes the hook and the last scene; a frame from the middle of each scene on the editor's own timeline
(`scene_frames`), 540 px wide JPEGs as base64 data URLs (no upload, no URL to fetch). A frame per scene
would cost ~1,000 more prompt tokens per extra image. The critic is told when the frames are placeholder
cards (it reads the prompt off the card) and is blind across rounds (it never sees its previous report or
what was changed: an evaluator that knows the fix tends to grade the fix).

**D10. Code measures, the model judges.** Word counts per scene and against the target, and whether the
hook was copied (difflib ratio >= 0.85 on normalised text, against the notes' suggestions and the ~200
example hooks in data/hooks), are computed and handed to the critic and the pick. A model counts words
badly; a copy is a string comparison, not an opinion.

**D11. Checkpoints: SQLite, `thread_id = run_id`, `durability="sync"`, an allowlist.** "sync" writes each
checkpoint before the next node starts; LangGraph's default "async" writes it while the next node runs, so
a hard kill can lose the last finished node. The serializer gets an explicit list of our classes: by
default LangGraph would import whatever class a checkpoint names (a writable checkpoint file = code
execution); with the list anything else comes back as plain data.

**D12. Approval = an interrupt in its own node, asked per paid clip.** `approve_hero` runs before every
producer pass but asks only when this plan would make a new clip (first pass, or a revision changed the
hero image); a "no" sticks for the run and switches that scene to the director's Ken Burns move, so the
video still finishes. `interrupt()` re-runs its node from the top on resume, so nothing before it may have
side effects (it only reads). The CLI never hangs: `--approve yes|no`, a prompt when stdin is a terminal,
otherwise the run pauses (state saved) and prints the resume command.

**D13. Tracing: autolog for LLM calls, our own span per node, never fatal.** `mlflow.openai.autolog()`
traces llm.py's OpenAI-SDK calls (Gemini included) with no change to llm.py; the node wrapper adds one
span per node with summaries (what the node read, what it returned, tokens, cost, seconds). MLflow
stores the critic's base64 frames as trace attachments (viewable in the UI, small span text). Every MLflow call is wrapped: a missing or
broken MLflow logs one warning and the run goes on (it happened for real in run 1, E1). Local store only:
`sqlite:///mlflow.db` + `mlruns/`; telemetry off with `MLFLOW_DISABLE_TELEMETRY=true` and `DO_NOT_TRACK=true`.

**D14. The ledger knows the run.** A context variable (`ledger.CURRENT_RUN`) set by the node wrapper tags
every ledger line with the run id, without adding a parameter to any llm/media function. "What did run X
pay for" is one filter, which is how the resume proof lists lines before and after.

**D15. Prompt versions in code, recorded per run.** `writer.PROMPTS = {"v1", "v2"}`; `RunState.prompt_version`
and `options.json` record which one wrote a run's script; MLflow runs log it as a param. v1 is exactly the
Phase 5 writer prompt + the options task; v2 changes only the writer (the research is shared in the evaluation).

---

## 12. The plan's check questions, answered from this build

**Q1. What does LangGraph give you that the Phase 3 loop didn't? And what does it cost?**

The Phase 3 loop is one agent: the model decides the next step (which tool), the state is the message
history, and everything lives in one process's memory. LangGraph is a different shape: the *program*
decides the steps (nodes and edges), the state is a typed object, and the framework runs the steps.
What that bought in this build, each one used and tested:
- **A typed, shared state with merge rules** (RunState + reducers): nodes return deltas; history is appended
  and usage summed without any node doing bookkeeping. The Phase 3 loop had one list of messages.
- **Control flow as data**: fixed edges for the known steps, a conditional edge for the critic loop, a
  conditional start for research. `get_graph().draw_mermaid()` draws it; the loop's control flow was a
  `while True` with `if`s.
- **Durable execution**: a checkpoint after every node in SQLite, `thread_id = run_id`. A 503 from Gemini
  (run 1), an injected `os._exit` (run 2) and a real `Stop-Process -Force` (run 2) all resumed without
  paying again (§5). The Phase 3 loop could only start over (or rely on tools reusing files).
- **Human-in-the-loop**: `interrupt()` pauses with the state saved; the process can exit; `Command(resume=...)`
  continues in another process, days later if need be (§6). In the loop that would be a custom pause-and-
  serialise mechanism.
- **History and inspection**: `get_state_history()` lists every checkpoint (state + next node); the 6.9
  measurement rebuilds each node's exact request from it.
- **Streaming** (`stream_mode="updates"`): the CLI's progress lines, for free.
- **A recursion limit** (25 steps by default) as a backstop against a loop that never ends.

What it costs:
- **Dependencies**: langgraph + langgraph-checkpoint + -sqlite + -prebuilt + -sdk + langchain-core
  (+ ormsgpack, xxhash, uuid-utils, langsmith, ...). One of them (uuid-utils, via langchain-core) didn't even
  load on this machine (E1). MLflow added 50 packages and a 13-minute download.
- **Abstraction**: nodes return partial dicts that are merged by channel rules; `invoke()` returns a dict,
  not your Pydantic model; fields never written are missing from that dict; `interrupt()` re-runs its whole
  node on resume (so no side effects before it); the default durability "async" can lose the last
  checkpoint on a hard kill. None of this is visible from the code of a node; it has to be learned.
- **Debugging**: a node's exception surfaces through LangGraph's runner frames (`runner.tick`,
  `run_with_retry`, `RunnableCallable.invoke`: see the 503 traceback in §4); the failed task is stored in
  the checkpoint (`tasks[0].error`); a hard kill leaves no error at all, only `next = producer`.
- **When not to use it**: for the Researcher or Producer alone (a straight line of known steps), a plain
  function is clearer. LangGraph pays off here because of the loop, the pause, and resuming.

**Q2. Why checkpoint after every node?**

Because a node is the unit of paid, slow work, and anything not checkpointed is redone on resume. On the
free tier one critic call took 363-473 s; a crash after it without a checkpoint costs those minutes (and,
on a paid tier, the tokens) again. Evidence from this build: the 503 crash in run 1's third review resumed
straight into `critic` with all 9 earlier ledger lines untouched; run 2 crashed before the director and
resumed into it (0 new lines for the researcher and writer). Within a node, progress is lost (a node is
atomic: run 2's producer was killed after 3 of 6 voices and ran again), which is why the producer itself is
idempotent (media.json) and why paid work is split into small nodes (the revision is its own node).
`durability="sync"` makes "after every node" literal: the checkpoint is on disk before the next node starts.
What it costs: a few milliseconds and ~kB per checkpoint (numbers in §5).

**Q3. What's the difference between a trace and a log? (Both for the same run.)**

A log is a flat, append-only list of events: `runs/ledger.jsonl` has one line per paid call (time, step,
model, tokens, cost, run_id), and the console shows one line per node. It answers "what happened and what
did it cost", is cheap, greppable, and survives anything (it's written before a response is even parsed).
A trace is a tree of spans with timing, parentage, inputs and outputs: the run's root span, one span per
node, and inside each node the exact LLM request and answer (autolog). It answers "why was this slow / what
exactly did the critic see / which node's prompt produced this", and how the parts relate. §7 shows the
same run both ways.

**Q4. How do you know prompt v2 is actually better?** (see §8)

---

## 13. Tests

`uv run pytest`: **323 passed** (235 before this phase + 88), ~25 s. Per file: test_ffmpeg 31, test_mcp_server 30,
test_graph **26**, test_tools 26, test_llm 24 (+7), test_critic **23**, test_youtube 23, test_loop 21, test_rag 20,
test_researcher 18, test_producer **14**, test_writer **12**, test_mcp_tools 11, test_ledger 10 (+2), test_tts 10,
test_models 9, test_make 7, test_images 4, test_toy_graph **4**. All offline: a fake OpenAI client that
answers by response-schema name and counts requests (so the real llm.py, ledger and validation run), fake
edge-tts, fake ffmpeg assembly and frame cutting, a real LangGraph with a real SQLite checkpointer in a temp
folder.
- **test_graph.py**: topic -> final video with one critic loop (scores, which scenes changed, `made` per
  round, the archived old picture, rejected option, calls per schema, usage per node from the reducer, ledger
  steps tagged with the run); the regenerated scene is among the next round's frames and the vision request
  has 4 `image_url` parts with data URLs; max rounds (3 reviews, 2 revisions, 3 edits); a good first score
  ends it; research first when asked (v2 notes reach the writer without ready-made hooks); routing (weak ->
  revise, passing score with a nitpick -> END, nothing flagged -> END, max rounds); `add_usage`; LangGraph's
  channel types per field; **crash + resume** (an injected crash before the director, a new graph on the same
  file, no repeated writer calls, the ledger's old lines unchanged and the first new one is the director's);
  a checkpoint per node in order; the serializer allowlist; **interrupt yes / no / pause-then-resume** and a
  revised hero image asking again; a span per node with tokens, summaries and the interrupt marked (fake
  MLflow); broken or missing MLflow never breaks a run; the CLI pause -> resume "no" -> "already finished";
  context_eval rebuilds each node's request byte for byte from the checkpoints; prompt_eval runs both
  versions on the same notes and the judge never sees the version.
- **test_critic.py**: frame picking (5 cases), frame times on the editor's timeline, the frame command, **real
  frames cut by ffmpeg** from a test video (540x960 JPEGs), the request's parts, script-only mode, code checks
  (words, copied hook), clamping, the schema has a list not a dict; the director (narration untouched, brand
  style wins, hero = ai_clip with the fallback, skipped shot, bad hero, style added once and only the missing
  parts); the reviser (only flagged scenes and fields, unchanged detection, the new hook, the shared style,
  its prompt); no docstring in the LLM schemas.
- **test_producer.py**: first pass makes everything and the second nothing; a new prompt remakes one image
  and archives the old one; new words remake one voice; a deleted file is remade; `clip_needed()` cases;
  `animate_image` needs the image and makes a clip once; the mock clip command and a **real 5-second mock
  clip**; the real fal path faked (price recorded before the download); ffmpeg `clip_cmd`; `assemble()` uses
  the clip for an ai_clip scene and Ken Burns when the clip is missing.
- **test_writer.py**: options validated one by one (word rule, language), one survivor skips the pick, the pick
  sees spoken words + facts and no image prompts, a pick off the list, all broken twice raises, v1 = the
  Phase 5 prompt + the task, v2's rules, v2's notes, unknown version, copied hooks (4 cases), the real library
  has ~200 examples.
- **test_toy_graph.py**: invoke returns a dict, the loop count, append vs replace, updates one node at a time.
- **test_llm.py** (+7): `chat_schema` doesn't validate; `paced` waits the 429's stated delay, gives up after
  the last retry, waits out 503s, raises at once on a daily quota; thinking tokens priced as output;
  OpenAI adds nothing. **test_ledger.py** (+2): lines carry the run id; pre-Phase-6 lines still parse.
- `uv run pytest exercises/phase-6`: 14 fail with `NotImplementedError` (by design); the same 14 pass against
  a reference solution (checked in a scratch folder). `uv run pytest exercises`: 64 fail (50 + 14), no clashes.

---

## 14. What it cost

From `runs/ledger.jsonl`, everything this phase sent (2026-10-23, 18:30-19:25 UTC; list
prices; **billed $0**: every call on Gemini's free tier):

| step | model | requests | input tokens | output tokens | list price |
|---|---|---:|---:|---:|---:|
| researcher | gemini-3.5-flash-lite | 9 | 13,057 | 2,979 | $0.011365 |
| writer (options) | gemini-3.5-flash-lite | 14 | 10,553 | 24,411 | $0.064193 |
| writer-pick | gemini-3.5-flash-lite | 14 | 9,735 | 2,191 | $0.008398 |
| director | gemini-3.5-flash-lite | 3 | 2,081 | 1,544 | $0.004484 |
| revise | gemini-3.5-flash-lite | 2 | 1,438 | 871 | $0.002609 |
| critic | gemini-3.8-flash | 4 | 17,869 | 847 | $0.016578 |
| critic (incl. 10 for the evaluation) | gemini-3.6-flash | 12 | 18,685 | 2,093 (+1,076 thinking) | $0.025898 |
| ctx-full-* (6.9) | gemini-3.5-flash-lite | 4 | 26,109 | 2,848 | $0.014952 |
| quota-probe | flash-lite, 3.6-flash | 2 | 8 | 4 | $0.000016 |
| rag (embeddings) | gemini-embedding-2 | 9 | 152 (est.) | 0 | $0.000030 |
| **total** | | **73** | **99,687** | **37,788** | **$0.148525** |

- **64 LLM calls** (plus 9 embedding requests), inside the ~120 budget: run 1 9 (the resumed critic included), run 2 5,
  run 3 5, the evaluation 35 (+3 in the stopped first attempt), vision probe 1, context 4, probes 2.
- Under-priced (E13): the Flash critic lines before the fix miss ~9,900 thinking tokens ≈ $0.037 list price.
- Not in the ledger: refused requests (429/503, not billed) and the YouTube quota: one live search (Chimgan,
  100 units, 0 results); every other search came from the 24-hour cache. Media $0 (placeholders, mock clips,
  edge-tts, ffmpeg).
- Ledger total for the whole project: $0.2148 list price.
- Wall time: the free tier's queue dominated. The 3.8 Flash critic took 6-8 minutes per call in run 1; the same
  kind of call took 9-37 s an hour later on 3.6 Flash.

---

## 15. Open items

- **The loop doesn't converge** (run 1: 5.5 -> 6.2 -> 6.0, new complaints each round). Options to try: give
  the critic last round's verdict on the *unchanged* scenes only, or ask it for a delta ("did the fix work?")
  as a separate check; sample the critic twice and average; or stop when the score stops rising.
- **Facts in revisions**: the reviser invented "Bon Cafe on Mirobod street" before the prompt rule; nothing
  checks for invented names automatically. A code check (proper nouns in a revision that appear nowhere in
  the topic, notes or brand rules) would catch it cheaply.
- **Length**: v2 put half its options in range, not all; the critic passes 25-second Uzbek videos (run 2: 50
  words). Either tighten `WORDS_SLACK` and retry with the measured count, or make the critic weigh length.
- **v3 of the writer prompt**: keep v2's no-copy and say/show rules, add "every scene gives the viewer one
  concrete fact, tip or step" (v1 won on concreteness), and discourage padding. Re-run the evaluation with 3
  samples per topic and a second judge, when there is quota (the evaluation needs ~20 critic calls).
- **Critic image tokens**: ~1,100 per 540x960 frame; Gemini's `media_resolution` (low) could cut that for
  placeholder cards; not tried through the compatible endpoint.
- **Producer spans inside MCP**: the MCP server's tool calls aren't traced (they run in another process);
  MLflow trace context could be passed along.
- **Director value**: after E5 it varies shot sizes and camera moves but mostly keeps the writer's subjects;
  maybe merge it into the writer, or give it a real job (continuity of people/places), when real images exist.
- **Thinking tokens before E13** remain under-priced in old ledger lines (append-only); a one-off
  correction note in the ledger is possible if exact totals matter.
- **Real media**: hero clip model id and price on fal.ai (never run), real images for a real visual critique.
- The Claude Desktop test from Phase 4 is still the owner's; the server now also offers `animate_image`.
