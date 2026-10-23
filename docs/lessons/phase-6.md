# Phase 6 — LangGraph + MLflow: the whole pipeline as a graph you can resume, approve and inspect

> Read at your own pace, ask anything. Parts 0–13 are the concepts. Parts 14–19 are what was really built
> and measured (from `docs/lessons/phase-6-build-notes.md`, commit 667af69): our code, real runs, the Critic
> loop, the crash-and-resume proof, the MLflow trace, the v1 vs v2 evaluation, errors, tests.

---

## 0. The problem: a straight line can't loop, pause, survive a crash, or explain itself

What we have so far:
- **`make` (Phase 2/5)** is a **pipeline**: plain Python calls research → write → images → voice → render,
  always in the same order. Simple and predictable, but:
  - it **can't go back** ("scene 3 is weak, redo only scene 3");
  - if it **crashes** at step 5, a re-run pays for steps 1–4 again;
  - it **can't pause** to ask you "this hero clip costs $0.30, continue?" and continue tomorrow;
  - when a video is bad, the only evidence is scattered files and the ledger.
- **The agent loop (Phase 3)** lets the **model** pick the order. Flexible, but slow on the free tier,
  harder to predict, and it still has no crash recovery or approval pause.

Phase 6 builds the full system the way companies build agent products:
```
topic ─► Researcher ─► Writer ─► Director ─► Producer ─► Editor ─► Critic ─┬─► END (good enough, or max rounds)
                                                 ▲                         │
                                                 └──── weak scenes only ◄──┘
```
- **The structure is decided by code** (a graph), like a pipeline.
- **One edge is decided by data**: the Critic's report decides whether to loop back.
- **The framework adds three things for free:** saving state after every step (crash → resume),
  pausing for a human (approval), and a clean place to hang tracing (MLflow).

---

## 1. Workflows vs agents: where a graph sits

Anthropic's "Building effective agents" guide names a few **workflow patterns** (code decides the path)
and contrasts them with **agents** (the model decides the path):

| Pattern | What it is | In Shortsmith |
|---|---|---|
| **Prompt chaining** | step A's output is step B's input | Researcher → Writer → Director |
| **Routing** | a step decides which branch comes next | after the Critic: loop or END |
| **Parallelization** | independent calls at the same time | 6 images, 6 voice lines (possible; today they run one by one) |
| **Orchestrator–workers** | one LLM splits work for others | (not used) |
| **Evaluator–optimizer** | one LLM generates, another judges, repeat | Critic → Producer loop |
| **Agent** | the model picks tools in a loop | Phase 3 producer (optional here) |

**The rule from Phase 3 still holds: use a workflow when the steps are known; use an agent only where the
path really can't be known in advance.** Our graph is mostly a workflow with one evaluator–optimizer loop.
That's not "less advanced"; it's what makes it cheap, testable and predictable.

---

## 2. LangGraph's core ideas: state, nodes, edges

LangGraph models a program as a **graph**:
- **State:** one shared object (a `TypedDict` or a Pydantic model) that flows through the graph.
- **Node:** a plain Python function `state → update`. It **returns only the fields it changed**, not the
  whole state. LangGraph merges the update into the state.
- **Edge:** "after node A, run node B". `START` and `END` are special nodes.
- **Conditional edge:** "after node A, call this **routing function** on the state; its answer picks the
  next node".
- **`compile()`** checks the graph (no missing nodes, no dead ends) and returns a runnable object.
- **`invoke(input)`** runs it until `END` (or a pause) and returns the final state.

A toy graph in the same shape as the one the build starts with:
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    draft: str
    score: int
    rounds: int

def write(state: State) -> dict:
    return {"draft": state["draft"] + " (revised)", "rounds": state["rounds"] + 1}

def judge(state: State) -> dict:
    return {"score": 4 + 3 * state["rounds"]}      # pretend: each revision scores better

def route(state: State) -> str:
    if state["score"] >= 8 or state["rounds"] >= 2:
        return "done"
    return "again"

builder = StateGraph(State)
builder.add_node("write", write)
builder.add_node("judge", judge)
builder.add_edge(START, "write")
builder.add_edge("write", "judge")
builder.add_conditional_edges("judge", route, {"again": "write", "done": END})
graph = builder.compile()

print(graph.invoke({"draft": "v0", "score": 0, "rounds": 0}))
# {'draft': 'v0 (revised) (revised)', 'score': 10, 'rounds': 2}
```
Three nodes' worth of ideas in 20 lines: state, two nodes, a normal edge, a conditional edge, a loop with
a stop rule.

**What a graph is *not*:** magic. Under the hood it's "run a node, merge its update, ask the edges what's
next, repeat". You'll write exactly that runner in plain Python in the exercise.

**"Standalone, without LangChain":** we use the `langgraph` package only. Our models are still called
through our own `llm.py` (so every call stays in the ledger). Honest detail: `langgraph` itself installs
`langchain-core` as a dependency (it uses some of its building blocks internally). We just don't use
LangChain's model wrappers, chains or prompts.

---

## 3. State reducers: how updates are merged

When a node returns `{"rounds": 2}`, the default is **overwrite**: the new value replaces the old one.
Sometimes you want something else:
```python
import operator
from typing import Annotated

class State(TypedDict):
    critic_rounds: Annotated[list[dict], operator.add]   # append, don't replace
    tokens_used: Annotated[int, operator.add]            # add up across nodes
    final_path: str                                      # plain: overwrite
```
- `Annotated[list, operator.add]` means "new list = old list + what the node returned". A node returns
  `{"critic_rounds": [report]}` and the report is **appended**.
- `Annotated[int, operator.add]` lets every node return its own `{"tokens_used": 812}` and the state keeps the
  **total**. No node needs to know the others' numbers.

**Why reducers exist:** with **parallel** nodes, two nodes may update the same field in the same step.
Overwrite would lose one of them; a reducer merges both. And for histories (every Critic report, every
round) appending is the natural rule. **A reducer is a merge policy per field.**

---

## 4. Our graph, node by node (and what each node is allowed to see)

| Node | Does | Reads | Writes |
|---|---|---|---|
| **Researcher** | trends + retrieved hooks → notes (Phase 5's `research()`) | topic, language, brand | research notes |
| **Writer** | 3 script options, then picks the best | topic, notes | options, chosen script |
| **Director** | final scenes + visual prompts, marks the hero scene | chosen script, brand style | final scenes |
| **Producer** | images, voice, (hero clip after approval) for the scenes that need it | scenes to make | file paths |
| **Editor** | ffmpeg assembly (Phase 2) | file paths, timings | `final.mp4` |
| **Critic** | samples frames + reads the script → scores, weak scenes, reasons | frames, script | a `CriticReport` |

**"Agents are just LLM calls with a role, a context and tools."** A "multi-agent system" here is not a
crowd of autonomous bots. It's a few functions, each with **its own prompt (role)**, **its own inputs
(context)** and sometimes **tools**. The Producer and Editor may not need an LLM at all.

**Writer: 3 options + self-pick** is a cheap version of **best-of-n**: generate several, select one.
- It helps because the first idea isn't always the best one.
- Honest limit: a model picking among its own options has **self-preference**; a different judge (as in
  Phase 5's experiment) is fairer. The real check is the Critic later.

---

## 5. The Critic: an LLM judge that can see

The Critic is an **LLM-as-judge** (Phase 5 §11) with **vision**:
1. **ffmpeg samples 3–4 frames** from `final.mp4` (e.g. one per scene in the middle of it).
2. The frames go into the chat request as **images** next to the script. In the OpenAI format a message's
   content becomes a list of parts:
   ```python
   {"role": "user", "content": [
       {"type": "text", "text": "Script: ... Rate each scene ..."},
       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}},
   ]}
   ```
   (A **data URL** puts the image bytes, base64-encoded, directly in the request; no upload needed.)
3. It returns a **validated `CriticReport`**: overall score 0–10, which scenes are weak, and why.

**Images cost tokens too.** A frame becomes hundreds of tokens, so we sample a few frames, small, not the
whole video.

**Honest limits:**
- In **mock mode** the frames are placeholder pictures, so "is this image beautiful?" can't be judged yet.
  The Critic can still judge the script, the narration–visual fit, and the known problems from Phase 5
  (word count, copied hooks, narration that describes pictures).
- **Judge scores are noisy and uncalibrated:** a 7 today may be a 6 tomorrow. That's why the loop needs a
  max-rounds rule, not "loop until 10/10".

---

## 6. The evaluator–optimizer loop = one conditional edge

```python
def route_after_critic(state) -> str:
    report = state.critic_rounds[-1]
    if report.score >= PASS_SCORE or not report.weak_scenes:   # good enough, or nothing to fix
        return END
    revisions_so_far = len(state.critic_rounds) - 1
    if revisions_so_far >= MAX_ROUNDS:                         # stop paying, even if not perfect
        return END
    return "revise"                                            # then producer: weak scenes only
```
(This is the shape of the real one in `agents/graph.py`: pass at 7, at most 2 revisions, so at most 3
reviews.)
Three design rules hide in those lines:
- **Only weak scenes are redone.** Regenerating everything would multiply cost by the number of rounds.
- **A redo must actually change something.** Regenerating with the same prompt usually gives a similar
  scene, so the Critic's *reasons* become a revised prompt or narration first.
- **A hard cap (max 2 rounds).** Without it, a picky judge and a stubborn generator can loop forever,
  burning money. **Every loop that costs money needs a stop rule in code** (same idea as Phase 3's
  `max_steps`).

---

## 7. Checkpointing: durable execution

A **checkpointer** saves the **whole state after every step** of the graph, under a **thread id** (for us:
the run id).
```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "2026-10-24_101500"}}
    graph.invoke({"topic": "..."}, config)   # crashes in the Producer...
    graph.invoke(None, config)               # ...later: continues from the last saved step
```
- `invoke(None, config)` means "no new input: **continue this thread** from its last checkpoint".
- `graph.get_state(config)` shows the saved state and **which node runs next**.

**Why after *every* node:** the expensive parts are in the middle (LLM calls, media). If state is saved
only at the end, a crash in the Editor throws away the paid Researcher, Writer and Producer work. Saved
after every node, **you lose at most the node that crashed.**

**What a checkpoint does *not* save:** the work *inside* the node that crashed. If the Producer made 4 of
6 images and then died, the Producer runs again on resume. So nodes should be **idempotent**: skip work
whose result already exists (an image already on disk isn't generated again).

**SQLite now, Postgres later:** SQLite is one local file, perfect for one user. In Phase 8–9, with several
users and a web server, the same graph would use LangGraph's **Postgres** checkpointer (the multi-user
question you asked earlier). Only the saver line changes.

---

## 8. Human-in-the-loop: `interrupt` and `Command(resume=...)`

```python
from langgraph.types import interrupt, Command

def producer(state):
    if state.wants_hero_clip:
        answer = interrupt({"question": "Hero clip costs about $0.30. Continue?", "cost_usd": 0.30})
        if answer == "yes":
            ...make the clip...
    ...

graph.invoke({"topic": "..."}, config)          # pauses: the result contains "__interrupt__"
graph.invoke(Command(resume="yes"), config)     # continues; interrupt() returns "yes"
```
- `interrupt()` **stops the graph, saves a checkpoint, and hands the question to the caller** (our CLI).
  It needs a checkpointer, because the pause can last seconds or days.
- **The big gotcha:** on resume, **the node runs again from its beginning**, and this time `interrupt()`
  returns the answer instead of pausing. So code *before* the `interrupt()` runs twice. Put expensive
  work **after** the approval, or make it idempotent.
- Where to pause matters: **right before money is spent** (the hero clip), never after.

---

## 9. Observability: logs, metrics, traces

| | What it is | Answers |
|---|---|---|
| **Log** | a line of text with a time ("Writer done in 41 s") | "what happened, roughly?" |
| **Metric** | a number over time (cost per video, average Critic score) | "is it getting better or worse?" |
| **Trace** | the **tree** of one run: run → node spans → LLM call spans, each with inputs, outputs, tokens, latency | "why did *this* video come out like this?" |

A **span** is one timed piece of work with a start, an end, inputs, outputs and attributes. Spans nest:
```
run "coffee shop in Tashkent"                    212 s   5,940 tokens
├─ researcher                                      38 s
│  └─ chat.completions (gemini-3.5-flash-lite)     35 s   2,104 tokens
├─ writer                                          61 s
│  ├─ chat.completions  (3 options)                44 s
│  └─ chat.completions  (pick)                     17 s
├─ director …
└─ critic                                          22 s   image ×4
```
(Illustrative numbers. The real trace comes from the build.)

**MLflow** is an open-source tool for this:
- `mlflow.openai.autolog()` makes **every OpenAI-SDK call** a span automatically. Our `llm.py` uses the
  OpenAI SDK even for Gemini, so Gemini calls are captured too.
- A **span per node** (a decorator or `with mlflow.start_span("writer"):`) builds the tree above.
- Everything is stored **locally** (a folder or a SQLite file), and `mlflow ui` opens a web page to browse it.

**Ledger vs trace:** the **ledger is the money truth** (one line per paid call, never skipped). The **trace
is the debugging story** (what each step saw and said). They overlap on tokens, but serve different jobs.

---

## 10. Evaluation and prompt versioning

"Prompt v2 is better" is a claim that needs evidence:
- **Version every prompt** (v1, v2) and record in each run which version it used. Otherwise you can't tell
  which prompt produced which video.
- **Same topics, same settings, only the prompt changes** (one variable at a time).
- **A judge that isn't the writer** (a different, stronger model), with a fixed rubric.
- **Log scores and cost per run** (MLflow), then compare: average score, wins per topic, cost per video.
- **Noise:** with 5 topics and one sample each, a difference of a point or two may be luck. A real claim
  needs more topics, repeated samples, or a big, consistent gap.
- **Goodhart's law:** if the rubric rewards "curiosity hooks", prompts will learn to produce curiosity
  hooks, not necessarily better videos. Read the outputs yourself too.

---

## 11. Context engineering, measured

Each node gets **only what it needs**:
- the **Editor** never sees research notes (it only needs files and timings);
- the **Critic** doesn't need raw trend data, only the script and the frames;
- the **Director** needs the chosen script, not all 3 options.

Why it matters: every extra token is **paid on every call**, slows the call, and gives the model more
chances to get distracted (Phase 5's coffee-shop loss came from notes leaking into narration). With traces
you can **measure** tokens per node before and after trimming, instead of guessing.

---

## 12. Hand-built loop (Phase 3) vs a framework (LangGraph)

| | Our Phase 3 loop | LangGraph |
|---|---|---|
| Control flow | a `while` loop; the model picks tools | an explicit graph; code picks most edges |
| State | a list of messages | a typed state with reducers |
| Crash recovery | none | checkpoint after every step |
| Human approval | would need custom code | `interrupt` + `Command(resume=...)` |
| Testing | easy (scripted fake LLM) | easy per node; graph tests need a bit of setup |
| Cost | ~150 lines, zero dependencies | a dependency (+ `langchain-core`), its abstractions, its version changes |
| Debugging | read our own code | sometimes read the framework's code |

**The honest answer for an interview:** "I built the loop by hand first, so I know what a framework does
underneath. I use LangGraph where it saves real work: durable state, resume and human approval. Those
are hard to get right by hand."

---

## 13. The commands (terminal, repo root)

```bash
uv run --env-file .env python -m shortsmith.make "coffee shop in Tashkent" --graph --research --approve yes
```
```bash
uv run --env-file .env python -m shortsmith.make --resume 2026-10-23_233023
```
- `--graph` runs the LangGraph pipeline (without it, the old Phase 2/5 pipeline runs, unchanged);
  `--research` and `--brand` work as in Phase 5.
- `--approve yes|no` answers the hero-clip question in advance. Without it you're asked `continue? [y/N]`
  in a terminal; with nobody to ask, the run **pauses, saves, and prints the resume command**.
- `--resume <run_id>` continues a crashed or paused run from its last checkpoint.
- Also: `--max-rounds` (2), `--pass-score` (7), `--prompt v1|v2` (v2), `--mcp` (media through the MCP
  server), `--no-trace` (no MLflow), `--crash-before <node>` (kills the process on purpose, for the demo).

---

## 14. Our code

**Versions installed:** langgraph 1.2.16, langgraph-checkpoint-sqlite 3.1.2, **langchain-core 1.7.0** (comes
with langgraph, as predicted in §2), mlflow 3.17.2. MLflow alone brought **50 packages** (pandas, pyarrow,
scikit-learn, flask…), a 13-minute download. That's part of a framework's price.

### 14.1 The real graph: 8 nodes, 2 more than the plan
```
  START ─(research on?)─► researcher ─┐
     └────────────────────────────────┴─► writer ─► director ─► approve_hero ─► producer ─► editor ─► critic
                                                                    ▲                                 │
                                                                    │   weak scenes, score < 7,       │
                                                                    └─────── revise ◄── rounds left ──┤
                                                                                                      ▼ otherwise
                                                                                                     END
```
| Node | LLM calls | What it does |
|---|---:|---|
| researcher | 1 | Phase 5's `research()` unchanged (optional: a **conditional edge from START**) |
| writer | 2 | **one** call writes 3 options, each validated alone, then **one** pick call |
| director | 1 | the picture side: one style, real shots (wide, close-up…), a camera move per scene, **which scene is the paid hero clip** |
| **approve_hero** | 0 | the `interrupt()`; asks only when a *new* paid clip would be made |
| producer | 0 | plain code: images, voices, the approved clip, **only what's missing or out of date** |
| editor | 0 | Phase 2's `ffmpeg.assemble()` (about 10 lines) |
| critic | 1 (vision) | 4 frames + the script + facts measured by code → `CriticReport` |
| **revise** | 1 | rewrites only the flagged scenes, only the fields the problem names |

**Why the two extra nodes:**
- **`revise` is its own node** (D7), because a Critic that also rewrites tends to grade its own fix. With its
  own node, the revision also gets its own checkpoint (a producer crash doesn't pay for the revision again)
  and its own trace span. The producer stays free of LLM calls.
- **`approve_hero` is its own node** (D12), because `interrupt()` re-runs its whole node on resume (§8). A
  node that only asks has nothing to repeat. `revise → approve_hero` (not straight to the producer): if a
  revision changes the hero image, the old clip is out of date, and a new paid clip must be asked again.

### 14.2 `RunState` and its reducers (`agents/state.py`)
| Fields | Reducer | Why |
|---|---|---|
| topic, language, settings, research, options, `plan`, `hero`, `final_path` | replace | one current value |
| `critic_rounds`, `revisions`, `made` | **`operator.add`** (append) | the whole history of the loop |
| `usage` (per node: calls, tokens, cost, seconds) | **`add_usage`** (a custom per-node sum) | every node reports only its own numbers |

- **Nodes return deltas, never totals** (D3). No node reads the total, adds its own and writes it back: that
  "read-modify-write" loses updates as soon as two nodes run in the same step. PLAN's `tokens_used` and
  `cost_usd` became properties computed from `usage`.
- **Two surprises of a Pydantic state**, measured on the toy graph:
  - `invoke()` returns a **plain dict**, not your Pydantic model (so the code does `RunState.model_validate(values)`);
  - fields never written are **missing** from that dict: defaults aren't stored values.
- A test checks LangGraph really made an appending channel for `critic_rounds` and a replacing one for `plan`.

### 14.3 The node wrapper (`node(name)` in `agents/graph.py`)
Every node is wrapped by one function, so the nodes stay small. The wrapper:
- **tags every ledger line with the run id** (a context variable, `ledger.CURRENT_RUN`, so no function needed
  a new parameter). "What did run X pay for?" becomes one filter. That's how §16's proof is made.
- can **crash the process on purpose** before a node (`--crash-before director` → `os._exit(70)`);
- opens an **MLflow span**: what the node read, a summary of what it returned, tokens, cost, seconds;
- treats an interrupt as **a pause, not a failure**.

### 14.4 The design choices that matter most
- **Writer: 2 calls, not 4** (D4). One call writes 3 options side by side (asked to use *different* hook
  patterns), for one prompt's price. Each option is **validated alone**: one option breaking the word rule
  doesn't throw away the other two. The pick reads only the **spoken words**, plus facts measured by code.
- **Producer = code, not an agent** (D6). The steps are known: every scene needs an image and a voice. Like
  `make`, it keeps a **manifest** (`media.json`: which prompt made each image, which words each voice says)
  and remakes only what's missing or changed. That one idea gives three things:
  - a revision remakes only the changed scenes, and only the changed part (new words → new voice, same image);
  - a crash inside the producer resumes for free;
  - replaced files go to `history/round_N/` instead of being deleted.
  
  Its tool calls go through `call_tool(name, json)`, so `--mcp` sends the very same calls through the MCP server.
- **Code measures, the model judges** (D10). Word counts and "was the hook copied?" (a string-similarity
  check against the notes and the ~200 example hooks in `data/hooks`) are computed by code and *handed* to the
  Critic and the pick. A model counts words badly; a copy is a string comparison, not an opinion.
- **The Critic sees 4 frames, chosen by a rule** (D9): just-regenerated scenes first, then an even spread that
  always includes the hook and the last scene, one frame from the middle of each scene, 540 px wide JPEGs as
  base64 data URLs. **It's blind across rounds:** it never sees its previous report or what was changed,
  because an evaluator that knows the fix tends to grade the fix.
- **The loop's rules** (D8): revise only when there are weak scenes **and** the score is under 7; at most 2
  revisions (so at most 3 reviews); at most 3 weak scenes per round ("half the video is weak" is a rewrite,
  not a fix).
- **Checkpoints** (D11): SQLite at `checkpoints/graph.sqlite`, `thread_id = run_id`, **`durability="sync"`**.
  - LangGraph's default ("async") writes a checkpoint *while the next node runs*, so a hard kill can lose
    the last finished node. "sync" makes "saved after every node" literally true.
  - A **class allowlist** for the serializer: by default LangGraph would import whatever class a
    checkpoint file names. A writable checkpoint file would mean **code execution**.
- **Prompt versions in code** (D15): `writer.PROMPTS = {"v1", "v2"}`. Every run records its version, and
  MLflow logs it as a parameter.

---

## 15. Real runs: nodes, tokens, latency, the Critic's rounds

### Run 1: the coffee shop, all the way to max rounds
```
run 2026-10-23_233023 | 'coffee shop in Tashkent, cozy autumn vibe' (en) | prompt v2 | max 2 revisions, pass at 7
researcher       4.3 s |   1477 in   349 out | notes: 3 hook patterns, 3 angles
writer          18.3 s |   1486 in  2183 out | 3 options, picked 1: 'Stop scrolling for the best Tashkent spot.'
director         2.4 s |    685 in   491 out | hero = scene 1: steam curls up from the surface of the warm latte ...
?? Scene 1 (...) as a 5-second AI clip costs ~$0.30 with real media; MOCK_MEDIA=1 makes a free placeholder clip now. Continue?
   answered by --approve: yes
producer        76.7 s |                     | round 0: images 1-6; voices 1-6; clip 1
editor          29.3 s |                     | final.mp4
critic         363.4 s |   5322 in   288 out | round 0: score 5.5, weak [5, 1, 3]
revise           5.2 s |    755 in   477 out | scene 1 narration; scene 3 narration; scene 5 narration
approve_hero     0.0 s |                     | no new clip needed
producer        22.9 s |                     | round 1: voices 1, 3, 5
critic         473.3 s |   5321 in   186 out | round 1: score 6.2, weak [1]
revise           2.1 s |    683 in   394 out | scene 1 narration
producer        12.0 s |                     | round 2: voices 1
   ... 503 from Gemini in the critic: the process exits ...
resuming 2026-10-23_233023 before critic
critic         500.9 s |   5318 in   213 out | round 2: score 6, weak [2, 5]
stop: max rounds (2 revisions; last score 6) | scores by round: [5.5, 6.2, 6.0]
final.mp4  1080x1920, 30 fps, 33.3 s
```
**Where it went:** 10 LLM calls, 25,645 tokens, $0.026 list price ($0 billed). **The Critic was 65% of the
tokens** (4 images × ~1,100 tokens each per review) **and 86% of the time** (6–8 minutes per call in the
free tier's queue. The same model answered in 9 seconds in run 3).

**What the loop changed:**
| Round | Score | The Critic's complaint | Before → after |
|---:|---:|---|---|
| 0 | 5.5 | scene 1: "overused cliché" | "Stop scrolling for the best Tashkent spot…" → "Step into the coziest autumn vibe with this rich spiced pumpkin latte." |
| 0 | | scene 3: "'happy mouth' feels unnatural" | "…melt instantly in your happy mouth." → "…melt in your mouth completely." |
| 0 | | scene 5: "claims an exact address without naming the venue" | "Save this exact address…" → "**Visit Bon Cafe on Mirobod street**…" ⚠️ |
| 1 | 6.2 | scene 1: "too generic"; fix: *"Is this the coziest autumn coffee spot in Tashkent?"* | the reviser used that line **word for word** |
| 2 | 6.0 | scene 5: "**name Bon Cafe earlier**" | (max rounds: stop) |

Only voices were remade (every problem was narration), so the images and the approved hero clip stayed.

**The mechanics worked:** only weak scenes went back, only they changed, the Critic looked at the regenerated
scenes first, and the run stopped at the limit.
**But it didn't converge** (5.5 → 6.2 → 6.0, new complaints every round), and it shows three real failure
modes of evaluator–optimizer loops:
1. **The optimizer hallucinated.** Asked to "name the specific cafe", the reviser **invented "Bon Cafe on
   Mirobod street"**, a name nobody gave it (and a real chain name, which is worse).
2. **The evaluator can't see a hallucination.** Next round the Critic accepted the invented name as a fact and
   asked to say it *earlier*. A judge of persuasiveness rewards confident specifics, true or not.
3. **The judge became the writer.** Its suggested hook was copied word for word, so the "writer" of that line
   is now the judge.

**Lesson: in a loop, the evaluator's suggestions are input to a generator, and they can push it into
inventing facts.** Fix: the reviser may not invent names, addresses, prices, dates or numbers; it makes a line
concrete another way or drops the promise. The useful part was real too: rounds 0–1 removed a cliché opening
and the "happy mouth" line.

### Does the Critic really see pixels?
In mock mode a frame is a coloured card with the prompt printed on it, and the Critic said so itself:
*"Actual visual quality, motion rendering, audio delivery, and pacing cannot be judged from text placeholder
cards."* So the build swapped one scene's image for a **real photo** (a Windows lock-screen picture: a lake at
sunset) under narration about "an armchair by a rainy window", and sent that one frame:
```
scene 4, "visual": "The rendered visual shows an outdoor sunset over water instead of an indoor cafe armchair."
```
It read the picture correctly and caught the mismatch. Still impossible until real images exist: image
quality, the same face or place across scenes, artefacts (hands, text in images), one consistent style, the
hero clip's motion.

### Runs 2 and 3
- **Run 2** (Registan, Uzbek) was the crash test (§16). It passed its first review at **7.8**, though the Critic
  noted it was short: 50 words, 25.2 s (target 55–60).
- **Run 3** (saving money as a student) ran in one go with **`--mcp`**: all 13 media calls (6 images, 6 voices,
  1 clip) went `producer → MCP client → stdio → mcp_server.py → tools`. **Score 8, no weak scenes, done in
  100 s.** A second pass over MCP made nothing (the manifest said "all up to date") in 2 s, which was just the
  server starting.

---

## 16. Crash + resume, and the approval step

**One checkpoint = 23.1 kB, 0.7 ms to write.** Run 1 left 18 of them (313 kB). Saving after every node is
almost free; the prompt is what's expensive (§17).

### Three crashes, nothing paid twice
**Crash 1, not planned: Gemini said 503** ("This model is currently experiencing high demand") in run 1's third
review. The OpenAI SDK's own 2 retries weren't enough, the node failed, the process exited. The checkpoint
said "before critic, round 2". The ledger for this run, filtered by run id, before the resume:
```
18:30:25 rag          gemini-embedding-2        17 in     0 out
18:30:27 researcher   gemini-3.5-flash-lite   1477 in   349 out
18:30:38 writer       gemini-3.5-flash-lite    793 in  2019 out
18:30:46 writer-pick  gemini-3.5-flash-lite    693 in   164 out
18:30:48 director     gemini-3.5-flash-lite    685 in   491 out
18:38:37 critic       gemini-3.8-flash        5322 in   288 out
18:38:43 revise       gemini-3.5-flash-lite    755 in   477 out
18:47:24 critic       gemini-3.8-flash        5321 in   186 out
18:47:26 revise       gemini-3.5-flash-lite    683 in   394 out
```
After `make --resume 2026-10-23_233023`: **the same 9 lines, plus exactly one new one**:
`18:58:46 critic gemini-3.8-flash 5318 in 213 out`. Nothing before the crash point was paid again.
(Fix for next time: `paced()` now also waits out a 5xx: 30 s, 60 s, 90 s.)

**Crash 2, injected:** `--crash-before director` kills the process with `os._exit(70)` right after the writer.
`os._exit` skips every `finally` and every flush, so only what's already on disk survives: the checkpoint after
the writer and 4 ledger lines. The resume started at the director; the researcher and writer cost 0 new lines.

**The pause:** that resume had no `--approve`, so the run reached the hero-clip question and **stopped with its
state saved**:
```
?? Scene 1 ("Kechki Registonga men bilan birga yur, ...") as a 5-second AI clip costs ~$0.30 with real media; ... Continue?
   continue? [y/N]
   no one to ask: the run is saved and paused here.
resume with:  uv run --env-file .env python -m shortsmith.make --resume 2026-10-23_235059 --approve yes|no
```
(Before a fix, this attempt crashed with `EOFError`. On Windows the "nothing" input device reports itself as a
terminal, so the CLI tried to ask. Nothing was lost: the director's work and the pause were already saved.)

**Crash 3, real:** a script started the resume with `--approve no`, watched the `audio/` folder, and killed
Python with `Stop-Process -Force` **as soon as 3 voice files existed**. On disk:
- images 1–3, voices 1–2 with their timings;
- `audio/scene_03.mp3` **with 0 bytes**: the kill landed while edge-tts was writing it.

The manifest had recorded only voices 1–2, because an entry is written **after** a tool returns, never before.
The resume remade **voice 3** and everything after it; the broken file went to `history/`. **A plain "does the
file exist?" check would have kept the empty file and broken the video.** That's why the producer trusts its
manifest, not the folder. The "no" wasn't asked again: `approve_hero` had finished and was checkpointed.

**What a checkpoint can't save:** a node killed halfway runs again from its start. Here that cost nothing,
because the producer has no LLM calls and skips finished files. That's the reason paid work sits in **small
nodes** and the producer is **idempotent**. (An LLM node killed mid-call loses that call: the answer never
reached the ledger, but the provider may still have counted it.)

### The approval step, in order
1. `approve_hero` calls `interrupt(question)` with the scene, the motion, the cost and "mock or real".
   LangGraph saves the state; `get_state().next` is `('approve_hero',)`.
2. The CLI's approver answers: `--approve yes|no` at once; a terminal asks `continue? [y/N]`; otherwise it
   pauses and prints the resume command. **The process can exit here.**
3. `Command(resume=answer)` re-runs `approve_hero` **from its first line**; this time `interrupt()` returns the
   answer.
   - **Yes:** the producer calls `animate_image` (a free mock clip while `MOCK_MEDIA=1`).
   - **No:** the hero scene falls back to the director's Ken Burns move, so the video still finishes, and you
     aren't asked again in that run.

The real fal path is written and tested with a fake, but **was never run**. Its model id and the $0.30
estimate must be checked on fal.ai before the first paid clip.

---

## 17. MLflow: one trace, read end to end; v1 vs v2; context trimming

### Setup
- `tracing.setup()` turns telemetry off (`MLFLOW_DISABLE_TELEMETRY=true`, `DO_NOT_TRACK=true`) **before**
  importing MLflow. Storage is **local**: `mlflow.db` and `mlruns/` in the repo, both gitignored.
- **`mlflow.openai.autolog()` does see Gemini.** It patches the OpenAI SDK, and `llm.py` uses that SDK with
  Gemini's address. It even caught the embedding request.
- The Critic's 4 frames are stored as trace **attachments**, so the UI shows the pictures.
- **Tracing never breaks a run:** run 1 started with MLflow failing to import (E1) and ran untraced with one
  warning line.

### Run 3's trace (15 spans, 99.3 s)
```
run 2026-10-24_001351 [AGENT] 99.32 s
  researcher    4.19 s  calls=1  in=1537  out=310
    Embeddings              1.07 s
    Completions             1.81 s  1537/310
  writer        6.44 s  calls=2  in=1439  out=1579        reads {"prompt_version": "v2", "notes_tokens": 251}
    Completions             4.52 s   795/1408     (3 options)
    Completions             1.55 s   644/171      (the pick: best = 3)
  director      4.98 s  calls=1  in=642   out=500         reads {the chosen script, "visual_style": ""}
  approve_hero  0.00 s                                     (the pause: interrupted = true)
  approve_hero  0.00 s                                     (the resume: approved = true)
  producer     48.70 s                                     images 1-6, voices 1-6, clip 1
  editor       25.57 s                                     final.mp4
  critic        9.23 s  calls=1  in=5444  out=123          score 8, no weak scenes
    Completions             7.92 s  5444/123 (total 6184)  4 frames as attachments
```
**What it shows at a glance:**
- **Where the time went:** once the free tier isn't queueing, the producer (49%) and editor (26%) dominate:
  edge-tts and ffmpeg, not the LLMs.
- **Where the tokens went:** the Critic's 5,444-token prompt is mostly 4 images.
- **What each node was given**, and the interrupt as two `approve_hero` spans.

**Reading this trace found three bugs** that no test had caught:
- **E13, hidden thinking tokens:** `total 6184` ≠ 5,444 + 123. Gemini's Flash models **think** before
  answering, and the compatible endpoint counts that thinking only in `total_tokens`, while Google bills it as
  output. **Every Flash call in our ledger was under-priced**: across 12 calls, **10,955 thinking tokens vs
  2,093 visible ones**. Fixed in the one place calls are priced: `reasoning = total − prompt − completion`.
  (MLflow's own cost estimate makes the same mistake.)
- **E15:** style words appended **twice** ("photorealistic, 8k resolution, photorealistic, 8k resolution").
- **E16:** **our docstrings were sent to the model**: Pydantic copies a class's docstring into its JSON schema,
  so a developer note went into the Critic's response format. Developer notes are comments now, and a test
  checks those schemas have no description.

**A crashed run's trace** stays `IN_PROGRESS` with only the spans that finished. **Tracing tells you what
happened up to a crash; the checkpoint is what makes it resumable.** Different jobs.

### Open it yourself (terminal, repo root, PowerShell)
```powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
$env:MLFLOW_DISABLE_TELEMETRY = "true"
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then open http://127.0.0.1:5000:
1. **Experiments → shortsmith → Traces**: one row per graph invocation. Click the one with
   `"topic": "how to save money as a university student"`.
2. Left pane: the span tree above. Click **writer** for its inputs, outputs and attributes (tokens, cost, seconds).
3. Click a **Completions** span under it: the exact system and user messages and the model's raw JSON. Under
   **critic → Completions**, "See more" shows the 4 frames the Critic saw.
4. **Experiments → shortsmith-prompt-eval → Runs**: the v1/v2 evaluation. Tick runs and press **Compare**.

Ignore the "MLflow Assistant" panel and "Detect issues" (they want their own LLM). The line "MLflow job backend
does not support Windows" is harmless. Stop the server with Ctrl+C.

### Trace vs log, for the same run
- **The log** (`runs/ledger.jsonl`, the 6 lines of run 3) answers **"what was paid, by which step, when"**.
  It's flat, append-only, written before a response is even parsed, and **survives every crash** (all of
  run 2's crashes left complete lines).
- **The trace** answers **"how did it unfold"**: which call belongs to which node, the 48.7 s of the producer the
  ledger can't see (no paid call), every node's inputs and outputs, and every full prompt.
- **The ledger is the money truth; the trace is where E13, E15 and E16 were spotted.**

### 6.8 Evaluation: Writer prompt v1 vs v2
- **v2's changes:**
  - exact words per scene, and "outside the range is thrown away";
  - "hook patterns are ideas, not sentences";
  - "the pictures SHOW, the narrator TELLS";
  - notes now carry patterns *without* a ready-made sentence, and angles split into say vs show.
- **The setup:** 5 topics (3 English, 2 Uzbek). Each topic gets **one shared research step**, then the Writer
  with v1 and with v2, then a **blind Critic** in script mode. The judge was gemini-3.6-flash, stronger than
  the writer; the order alternated per topic. 35 calls, $0.

| | v1 | v2 |
|---|---:|---:|
| **mean Critic score** | **7.40** | **7.40** |
| topics won | 3 | 2 |
| **options with a copied hook** (measured by code) | **11 / 14** | **0 / 14** |
| **options inside the word target** | 2 / 14 | **7 / 14** |
| chosen scripts inside the target | 1 / 5 | 3 / 5 |
| "narration describes the picture" flagged | once | never |
| Writer cost (list price) | $0.0251 | $0.0265 (+5%) |

**Reading it, the build's and mine:**
- **On what v2 targeted, it clearly worked**, and those numbers are measured by code, not by a judge: copied
  hooks 79% → **0%**, 3.5× more options inside the word target.
- **On overall quality there's no difference: 7.40 vs 7.40**, a 2–3 split. With one sample per topic and
  differences of 0.5–1.5 points, that's what "no difference" looks like (Phase 5 saw the same judge move a
  script by up to 2 points between runs).
- **Per topic:**
  - coffee shop: v2 clearly better. v1's "Golden autumn light streams through large windows…" is a caption,
    not narration.
  - saving money: v1 better; it's concrete ("ask cashiers if they have a student deal"), v2 is stiff.
  - plov: v1 better. v2 says "use **suzma** instead of ordinary rice". Suzma is strained yogurt, so that's
    nonsense, and the Critic caught it.
- **Length still isn't solved:** half of v2's options are a few words short.
- **Decision: v2 stays the default.** Copied hooks (two videos opening with the same line) and wrong lengths
  (a 23-second video) are **defects regardless of taste**.
- **What would make "v2 is better" a real claim:** 20+ topics, 3 samples each, a second judge or human
  raters, and a paired test on the score differences.

**How do we know v2 is better?** It's better **at what it was written for** (measured). We **don't** know it
writes better videos. Saying exactly that is the honest answer.

### 6.9 Context trimming, measured
Each node's exact request was rebuilt from run 1's checkpoints, once as sent (trimmed) and once with the
**whole state** appended (the common shortcut: give every agent everything). Gemini's own token counts:
| Node | Trimmed | Whole state | × |
|---|---:|---:|---:|
| writer (options) | 793 | 4,707 | 5.9× |
| writer-pick | 693 | 6,704 | 9.7× |
| director | 685 | 6,741 | 9.8× |
| revise | 755 | 7,957 | **10.5×** |
| critic | 5,322 | ~12,100 (estimated: the daily quota was gone) | ~2.3× |

- **The four text nodes: 2,926 tokens instead of 26,109 per run.**
- **The state grows during a run** (options, reports, revisions), so the late nodes pay the most: revise 10.5×,
  and it runs up to twice per video.
- **The Critic is dominated by images**, so trimming its text saves less. The lever there is fewer or smaller frames.
- **Fewer tokens is only half of it:** a node that sees the notes can act on them, which is how Phase 5's
  coffee-shop loss happened. This build measured the tokens, not that quality effect.

---

## 18. Errors that happened, and what they teach
| # | What happened | Lesson |
|---|---|---|
| E1 | `ImportError: DLL load failed … An Application Control policy has blocked this file` on the first `import langgraph` | Windows **Smart App Control** blocks brand-new native binaries with no reputation (langchain-core 1.7.0 was 2 days old). Fixed by **pinning older versions** that load (`[tool.uv] constraint-dependencies`), **never by touching your security settings** |
| E2 | `draw_ascii()` needs another package | use `draw_mermaid()` (paste into mermaid.live) |
| E3 | `SyntaxWarning: invalid escape sequence` from a docstring diagram | a backslash in a normal string is an escape |
| E4 | six test bugs, e.g. a test that passed against the *empty* exercise | `NotImplementedError` **is a** `RuntimeError`: check that a test can fail |
| E5 | the Director handed the writer's image prompts back unchanged | found by **looking at a frame**; its prompt now says "rewrite the idea as a shot" |
| E6 | the reviser **invented "Bon Cafe on Mirobod street"** | the evaluator's fix can push the generator to hallucinate; forbid inventing facts |
| E7 | a revised hook cut mid-phrase by a 10-word truncation | never truncate language with code; ask the model for the hook itself |
| E8 | MLflow prints an "agent hint" on import | turn noise off (`MLFLOW_DISABLE_AGENT_HINT=1`) |
| E9 | `503 … high demand` crashed the Critic | **the checkpoint saved the run**; `paced()` now waits out 5xx too |
| E10 | `EOFError` on resume: Windows' NUL device says "I'm a terminal" | "nobody to ask" must pause, not crash |
| E11 | **`429 … limit: 20, model: gemini-3.8-flash … PerDay`** | the big one: **each free Flash model allows 20 requests a day**. The judge's daily quota, not money, is the evaluation budget. Critic model made configurable (`SHORTSMITH_CRITIC_MODEL`); a per-day 429 now stops at once instead of retrying |
| E12 | the Critic flagged "scene 1 repeats the hook" | it saw a separate `Hook:` line; **how you show data to a judge changes its verdict** |
| E13 | **thinking tokens missing from the ledger** | found by reading a trace; "compatible" ≠ identical, again |
| E14 | a prompt's example came back as output ("steam curls up… off the student ID") | models copy examples; describe the shape of the answer instead |
| E15 | style words added twice | found in the trace |
| E16 | developer docstrings sent to the model inside the JSON schema | a schema is a prompt: everything in it reaches the model |

**Open items** (for later phases):
- length is still missed sometimes;
- the Critic doesn't converge across rounds (blind, noisy);
- a fact-check for revisions;
- run voices and images in parallel;
- fewer or smaller frames for placeholder cards;
- v3 of the writer prompt (v1's concreteness without padding);
- check the fal model id and price before the first real hero clip.

---

## 19. Tests, and your exercise 🧠

- **323 tests** (+88), ~25 s, all offline:
  - test_graph 26, test_critic 23, test_producer 14, test_writer 12, test_toy_graph 4, plus additions in
    llm/ledger/mcp.
  - The fake OpenAI client **answers by response-schema name and counts requests**, so the real `llm.py`,
    ledger and validation run.
  - A **real LangGraph with a real SQLite checkpointer** runs in a temp folder.
- **What they prove:**
  - topic → video with a Critic loop (only the weak scene changes; it's among the next round's frames; the
    vision request has 4 images);
  - max rounds; a good first score ends it; every routing rule;
  - **crash + resume** with no repeated writer calls and the old ledger lines unchanged;
  - **interrupt yes / no / pause-then-resume**;
  - a revised hero image asks again;
  - a broken MLflow never breaks a run;
  - each node's request rebuilt byte for byte from checkpoints;
  - the judge never sees the prompt version.
- **🧠 Your exercise:** `exercises/phase-6/graph_runner.py`. What LangGraph does for us, in plain Python, with
  plain dicts you can check on paper:
  1. `merge_update(state, update, reducers)`: how a node's partial update is merged (replace, append, sum);
  2. `route_after_critic(state, pass_score, max_rounds)`: the loop's exit, exactly the rules of §6;
  3. `run_graph(nodes, edges, start, state)`: run a node, merge, follow a fixed or conditional edge, stop at
     END, and raise after 25 steps (LangGraph's **recursion limit**).
  ```
  uv run pytest exercises/phase-6
  ```
  14 tests, all red now. When they pass, you've written the core of a graph framework yourself.

---

## Check questions (optional, answer whenever you like)
1. What's the difference between a pipeline, a graph and an agent? Which one is Shortsmith now, and why?
2. A node returns `{"tokens_used": 500}`. What happens with and without `Annotated[int, operator.add]`?
3. Why must the Critic loop have a max-rounds rule even if the judge were perfect? And since it isn't?
4. Why does the Critic's reason need to reach the Producer, instead of just "redo scene 3"?
5. Why checkpoint after every node and not only at the end? What does a checkpoint *not* protect?
6. On resume after an `interrupt`, the node runs again from the top. What bug can that cause, and how do
   you avoid it?
7. Trace vs log: what can you answer with a trace that you can't with a log?
8. How would you convince a skeptical teammate that prompt v2 is better? What could fool you?
9. What does LangGraph give you that the Phase 3 loop didn't? What does it cost?
10. Why does the Editor never see research notes?

**Our build**
11. Why are `revise` and `approve_hero` separate nodes instead of parts of the Critic and Producer?
12. Why must nodes return deltas (their own tokens) and never totals?
13. What does `durability="sync"` change, and why does the checkpoint serializer need a class allowlist?
14. Run 2's kill left a 0-byte voice file. Why did the resume remake it, and why would "does the file
    exist?" have been a bug?
15. In run 1, how did the evaluator–optimizer loop produce an invented cafe name? What fixed it?
16. The Critic's request showed `total 6184` for 5,444 + 123 tokens. What was missing, and why did it matter?
17. v1 and v2 both scored 7.40. Why does v2 stay the default anyway, and what can't we claim?
18. Why does the Critic see 4 frames, chosen by a rule, and never its previous report?
19. On the free tier, what limited the evaluation more than money?
20. Trace vs ledger: name one thing only the trace showed, and one thing only the ledger guarantees.

## Interview angles
- "My pipeline is a LangGraph graph: a workflow with one evaluator–optimizer loop, where a vision Critic
  sends only weak scenes back, capped at 2 rounds."
- "Every node is checkpointed, so a crashed run resumes without paying again, and an `interrupt` asks for
  approval before the one expensive step."
- "Every run is an MLflow trace with a span per node, so I can see each step's prompt, output, tokens and
  latency, and I compare prompt versions with logged scores and cost."
- "I wrote the agent loop by hand before using a framework, so I know what LangGraph does underneath and
  where it's worth the dependency."
- "I proved resume with the ledger: a real 503, an injected crash and a hard process kill all resumed with
  zero repeated paid calls, because paid work sits in small checkpointed nodes and the producer is idempotent."
- "My evaluator–optimizer loop taught me its failure mode: the critic asked for a specific name, the reviser
  invented one, and the critic accepted it, so the reviser is now forbidden to invent facts."
- "Reading one trace found three bugs no test caught, including Gemini's thinking tokens missing from usage,
  which made every Flash call under-priced."
- "Prompt v2 removed copied hooks completely (79% → 0%) and tripled options inside the length target, but the
  blind judge scored both 7.40, so I claim exactly that and no more."
- "Context trimming cut the text nodes' prompts 6–10×: 2,926 tokens per run instead of 26,109 with the whole state."
