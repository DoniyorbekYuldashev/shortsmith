# Phase 3 — Tool use from scratch: what an "AI agent" really is

> Read at your own pace, ask anything. Parts 1–9 are the concepts (they don't depend on our code).
> Parts 10+ (our code, the real message history, numbers, errors, tests) are completed from the
> build session's notes once the build is done.

---

## 0. The big picture: pipeline vs agent

In Phase 2, **our code decided everything**: `make.py` always runs script → images → voice → ffmpeg,
in that order, every time. That's a **pipeline** (also called a *workflow*).

In Phase 3, the **model decides** what to do next. We give it a goal ("turn this script into a
video") and a set of **tools** (generate an image, make a voice file, render the video). It looks at
where things stand and picks the next tool, again and again, until it says "done".

```
Pipeline (Phase 2)                       Agent (Phase 3)
──────────────────                       ───────────────
code: images()                           model: "call generate_image(scene 1)"
code: voice()                            code:  runs it → "ok, scenes/scene_01.png"
code: assemble()                         model: "call text_to_speech(scene 1)"
                                         code:  runs it → "ok, 4.8 s"
                                         ...
                                         model: "call render_video()"
                                         code:  runs it → "ok, final.mp4"
                                         model: "Done: the video is ready."
```

**An agent = an LLM + tools + a loop.** No magic. That sentence is the whole phase.

---

## 1. Function calling (tool use): who does what

The single most important fact, and a classic interview question:

> **The model never executes anything.** It only *writes a request*: "please call tool X with
> these arguments". **Your code** reads that request, runs the real function, and sends the result
> back as a message.

The model is a text generator. Function calling is a special *output format*: instead of prose, the
API returns a structured **tool call**:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": { "name": "generate_image", "arguments": "{\"scene\": 1}" }
  }]
}
```

Your code runs `generate_image(scene=1)` and answers with a **tool message**:

```json
{ "role": "tool", "tool_call_id": "call_1", "content": "{\"ok\": true, \"path\": \"scenes/scene_01.png\"}" }
```

Notes:
- `arguments` is a **JSON string**, not an object: the model *wrote* it, so it can be malformed or
  wrong, and your code must validate it (that's where Pydantic comes in again).
- `tool_call_id` links each result to the call it answers. That matters when there are several
  calls at once (section 6).
- The message roles are now **four**: `system` (rules), `user` (the task), `assistant` (the model's
  turns, text and/or tool calls), `tool` (results from your code).

Analogy: the model is a **manager on the phone**. It can't touch anything in the workshop; it can
only tell the worker (your code) "cut this piece", and the worker reports back "done, here it is".

---

## 2. Tool schemas: how the model knows what tools exist

With every request you send the list of tools as **JSON Schemas**, the same kind of schema as the
structured outputs in Phase 1 (Step 1.4):

```json
{
  "type": "function",
  "function": {
    "name": "generate_image",
    "description": "Create the picture for one scene of the script. Call once per scene.",
    "parameters": {
      "type": "object",
      "properties": { "scene": { "type": "integer", "description": "scene number, 1..6" } },
      "required": ["scene"]
    }
  }
}
```

- **The description is a prompt.** The model chooses tools by reading names and descriptions, so
  clear names, "when to use it" and "what it returns" matter as much as the code.
- **Few, focused tools** beat many overlapping ones: every schema costs tokens on every call, and
  similar tools confuse the model.
- **Small results.** Every tool result is sent back to the model and stays in the history. Return
  `{"ok": true, "path": "..."}`, never a whole file.
- **Generate the schema from a Pydantic model** (like `chat_json` did for the Script): the schema the
  model sees and the validation your code runs can never disagree.

---

## 3. The agent loop

The whole idea in a few lines of pseudo-code:

```
messages = [system prompt, user task]
for step in range(max_steps):
    reply = llm(messages, tools)          # one paid call
    messages.append(reply)                # the assistant turn, exactly as returned
    if reply has no tool calls:
        return reply.text                 # the model says it's done
    for call in reply.tool_calls:
        result = run_tool(call.name, call.arguments)   # YOUR code executes
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
stop: "max_steps reached"
```

What happens on each turn:
1. The **whole history** goes to the model (it's stateless, Phase 1).
2. The model reads it and either asks for tools or answers in plain text.
3. Your code runs the tools and appends the results.
4. Repeat.

You'll write this loop yourself in the exercise. In an interview, being able to write it from memory
is a strong signal that you understand agents, not just frameworks.

---

## 4. Stop conditions: why `max_steps` exists

A loop that only stops when the model says "done" can run forever. Real reasons it does:
- the model keeps retrying a tool that always fails;
- it repeats the same call because it "forgot" it already succeeded;
- a prompt injection tells it to keep going.

Each turn is a paid call, and the history (and so the cost per turn) keeps growing (section 7).
So agents need **hard limits**, enforced by *your code*, not by the model:
- **max_steps**: a maximum number of LLM turns;
- **token budget**: stop when the total tokens pass a limit (the ledger already counts them);
- (later) **cost budget and wall-clock time**: the `--budget` flag from the plan.

Stopping must be **clean**: record why it stopped (`"done"`, `"max_steps"`, `"token_budget"`) so
you can tell a finished run from a cut-off one.

---

## 5. Errors as feedback

In normal code, an exception stops the program. In an agent, a failing tool should usually **not**
crash the loop. Instead, the error goes back to the model as the tool result:

```json
{ "role": "tool", "tool_call_id": "call_7", "content": "{\"ok\": false, \"error\": \"scene 9 does not exist (1-6)\"}" }
```

The model reads it like a human would and **adapts**: fixes the argument, tries another tool, or
explains why it can't finish. This is one of the biggest differences between agents and pipelines:
the error message becomes **input for the next decision**.

- Write error messages **for the model**: say what was wrong and what's allowed.
- Unknown tool names, invalid JSON arguments and failed validation are all handled the same way.
- But there's a limit: if the same thing fails again and again, `max_steps` ends it. Some systems
  also stop after N consecutive errors.

---

## 6. Parallel tool calls

A model can ask for **several tools in one turn**, e.g. "generate images for scenes 1–6" as six calls
in one assistant message. Your code then:
- runs all of them (one after another, or truly in parallel);
- appends **one tool message per call**, each with its matching `tool_call_id`.

Why it matters: **fewer turns = fewer paid calls and a shorter history**. Six images in one turn is
one LLM call instead of six. Whether the model does it depends on the model and the prompt; the build
notes will show what Gemini actually did.

---

## 7. The context grows every turn

Because the model is stateless, turn 10 resends turns 1–9: the system prompt, the task, every tool
call and every tool result. So **prompt tokens grow every turn**, and the total cost of a run grows
faster than the number of turns (roughly with its square, if every turn adds a similar amount).

That's why:
- tool results must be **short**;
- the task should contain only what the agent needs (a compact script, not everything);
- long-running agents **trim or summarise** old turns (context engineering, Phase 6.9).

The build notes will show the real numbers: prompt tokens per turn, climbing.

---

## 8. Safety: tools are the only actions

- The model's text is **never executed** as code or shell commands. It can only pick from the tools
  you defined, and your code decides what each tool is allowed to do.
- **Validate every argument** (Pydantic): the model may invent scene 9 or pass a string.
- **Least privilege:** our tools are bound to one run folder and take scene numbers, not raw file
  paths, so a confused or manipulated model can't touch other files.
- This is where Phase 1's prompt-injection lesson becomes real: an agent that **acts** is far riskier
  than one that only writes text.

---

## 9. Honest trade-off: when to use an agent at all

Our video pipeline is naturally a **fixed sequence**. A pipeline is cheaper (no LLM call per step),
faster, and fully predictable. So why build an agent?
- **Flexibility:** it can retry a failed scene, skip work that's already done, or change order
  without anyone coding those cases.
- **It's the foundation** for Phase 4 (the same tools over MCP, usable from Claude Desktop) and
  Phase 6 (several agents and a critic).
- **Learning:** it's the clearest way to see what frameworks like LangGraph do for you later.

The rule of thumb used by people who build these systems: **use a workflow when the steps are known,
an agent when they aren't.** Saying that in an interview, and knowing that our video assembly is
mostly a workflow, shows judgment, not just enthusiasm.

---

## 10. Our code

Three files: `llm.py` (one new function), `agents/tools.py` (the tools), `agents/loop.py` (the loop).
**No agent framework and no new dependency**: the loop itself is ~40 lines of plain Python.

### 10.1 `llm.py`: tool calling added
- **`Message = dict[str, Any]`**: one chat message in the API's own format. On purpose a plain dict,
  not a Pydantic model: Gemini adds fields of its own to assistant messages (`extra_content`, see
  section 13, E1) that must go back unchanged, and a typed model would drop them.
- **`ToolCall(id, name, arguments)`**: one call the model asked for. `arguments` stays a JSON
  **string**: it's the model's output, so it's validated later, never trusted.
- **`ChatResult`** got two fields: `tool_calls` (parsed, for our code) and **`message`** (the assistant
  message *exactly as returned*, for the history).
- **`chat_tools(messages, tools, ...)`** is one agent turn. **It runs nothing**: it returns what the
  model asked for. It still goes through `_complete()`, so every turn is priced and in the ledger.
  The tool schemas are sent with every turn and count as prompt tokens each time.

### 10.2 `agents/tools.py`: the tools
**`ToolContext`**: what every tool call shares, and **the model never sees**: the run folder, its
script, the voiceovers made so far, and the demo switch for injected failures. This is how the
*program* binds one run folder to every call (least privilege, section 8).
`ctx.scene(9)` raises `ValueError("no scene 9: this script has scenes 1-6")`, an error written so the
model can understand it.

**The registry and the `@tool` decorator**
```python
TOOLS: dict[str, Tool] = {}

@tool(SceneArgs)
def generate_image(ctx: ToolContext, args: SceneArgs) -> ImageResult:
    """Make the vertical 9:16 image for one scene, from the script's visual prompt.
    Saves scenes/scene_NN.png. An image that already exists is reused, not made again."""
```
- A **decorator** is a function that wraps another function at definition time. `@tool(SceneArgs)`
  registers `generate_image` in `TOOLS` with: name = the function's name, **description = its
  docstring** (`inspect.getdoc`), arguments model, and the function itself.
- The docstrings here are written **for the model**: what the tool does, what it saves, what it needs
  first. Phase 4's FastMCP `@mcp.tool()` does exactly the same, so moving to MCP is mostly re-registering.

**Arguments: what the model may send**
```python
class SceneArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene: int = Field(ge=1, description="Scene number (idx) from the script, starting at 1.")
```
- `Field(ge=1, ...)` becomes `"minimum": 1` + a description in the schema; `extra="forbid"` becomes
  `"additionalProperties": false`; no default makes it `"required"`.
- The **same model** both generates the schema the API sees and validates the model's answer
  (`SceneArgs.model_validate_json(arguments)`), so they can't disagree.
- `NoArgs` for `render_video`: an empty object.

**Results: what goes back (short!)**: `ImageResult(scene, path, reused)`,
`VoiceResult(scene, path, duration_s, words)`, `VideoResult(path, duration_s, size_mb, scenes)`.
Paths are relative to the run folder. A typical result is 72 characters.

**The three tools**
| Tool | What it does | Why designed this way |
|---|---|---|
| `generate_image(scene)` | reads that scene's `visual_prompt` from the script, makes `scenes/scene_NN.png` (placeholder now) | an existing image is **reused** (`reused: true`), so a retrying model never pays twice |
| `text_to_speech(scene)` | reads that scene's narration, makes `audio/scene_NN.mp3` with edge-tts | once every scene has a voice, it saves `voiceovers.json` so `make --run` can re-edit later |
| `render_video()` | Phase 2's `ffmpeg.assemble()`, unchanged | **first checks** everything exists; if not, a precise error: `not ready to render: no image for scenes [2, 3]; no voiceover for scenes [1]` |

**Why tools take scene numbers, not text or paths** (decision D1):
1. The script stays the **single source of truth**: the model can't "improve" an Uzbek narration and break the subtitles.
2. Arguments are tiny: `{"scene":2}` is 5 characters instead of a 30-word prompt. Every argument is paid **twice**: as output now, as input on every later turn.
3. No tool takes a path, so nothing the model writes can point outside the run folder.

**Schemas**: `schema(tool)` builds the `{"type": "function", "function": {name, description, parameters}}`
format. `_without_titles()` removes Pydantic's automatic `"title"` keys ("SceneArgs"…): they tell the
model nothing, and removing them saves **33 tokens on every turn** (338 → 305 for the three tools).

**`call_tool(ctx, name, arguments) -> str`**: the gatekeeper, in order:
1. unknown name → `ValueError("unknown tool 'x'; the tools are: generate_image, text_to_speech, render_video")`;
2. validate the arguments (bad JSON, wrong type, `scene: 0`, extra fields → `ValidationError`);
3. the injected demo failure, if switched on;
4. run the function;
5. return `{"ok": true, ...}` as JSON.

It **raises** on any problem. Turning errors into feedback is the loop's job (next section).

### 10.3 `agents/loop.py`: the loop
**Dependency injection** (a key design idea): the loop doesn't import the LLM or the tools. It *receives*
them as functions:
```python
CallLLM  = Callable[[list[Message], list[Message]], ChatResult]   # (messages, tool schemas) -> answer
CallTool = Callable[[str, str], str]                              # (name, JSON arguments) -> result
```
`main()` passes the real ones; the tests pass **scripted fakes**. The same loop runs both, and in
Phase 4 an MCP client simply becomes another `CallTool`.

**`run_loop(...)`, the heart of the phase** (simplified):
```python
while True:
    result = call_llm(messages, tool_schemas)          # one paid turn
    used += result.total_tokens
    messages.append(result.message)                    # UNCHANGED (thought signatures, section 13)
    for call in result.tool_calls:                     # parallel calls: run all, answer each
        try:
            content, ok = call_tool(call.name, call.arguments), True
        except Exception as e:                         # errors are feedback, not crashes
            content, ok = error_result(e), False
        messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    stop = "done" if not result.tool_calls else \
           "token_budget" if used >= max_tokens else \
           "max_steps" if this was turn number max_steps else None
    log(Turn(...))
    if stop: return AgentResult(...)
```
Details that matter:
- **One `try/except` in one place** guarantees that *no* tool (including future MCP tools) can crash
  the loop. `KeyboardInterrupt` isn't an `Exception`, so Ctrl+C still stops a run.
- **Stop rules, checked after each turn, in this order** (D9):
  - `done` wins: the model answered without tool calls, so the job is finished, even if over budget;
  - `token_budget`: checked *before* paying for another call, so it can overshoot by at most one turn;
  - `max_steps`.
- **The last turn's tool calls still run** before stopping, so every tool call in the history has its
  answer and the conversation is valid to continue later.

**`error_result(e)`**: an exception → `{"ok": false, "error": "..."}`, at most 300 characters:
- a `ValidationError` → `invalid arguments: scene: Input should be a valid integer` (no long Pydantic URL);
- an ffmpeg failure → `ffmpeg failed with exit code 1 | ffmpeg said: ...` (not the 1,500-character
  command that `str(e)` would print).

The model reads this, and pays for it on every later turn.

**Records**: `CallLog` (one tool call: name, arguments, result, ok, seconds), `Turn` (one line of
`agent_log.jsonl`: tokens, tokens so far, cost, LLM latency, stop reason), `AgentResult` (the summary
and the whole conversation).

**`llm_caller(...)`: free-tier manners**, kept *outside* the loop (D10):
- requests start **at least 4.5 s apart** (~15 per minute allowed);
- a **429 / RESOURCE_EXHAUSTED** waits 20, 40, then 60 s and retries, giving up on the 4th attempt;
- a refused request isn't billed, so only successful calls reach the ledger;
- `sleep` is a parameter, so the tests don't really wait (the same trick as Phase 2's retry tests).

**`PRODUCER_SYSTEM_PROMPT`**: it describes **the job, the one dependency and when to stop**, *not* the steps:
```
You are the producer of a short vertical video. The script is already written; your job is to turn it into final.mp4 using your tools.
- Every scene needs an image and a voiceover before the video can be rendered.
- You may ask for several tool calls at once when they don't depend on each other.
- Tools answer with JSON. If one returns "ok": false, read the error and decide: retry, fix the cause, or stop.
- When final.mp4 is rendered (or it clearly can't be), reply with one short sentence and call no tool.
```

**`task_message(ctx)`**: the user message, kept compact: title, language, each scene's narration
(**no visual prompts**, since the tools read those themselves), plus what already exists
("Already made: images for scenes [...]"), so a half-finished run isn't redone.

**`journal(path)`**: appends each `Turn` to `agent_log.jsonl` **as it happens** (a crash keeps the
turns so far) and prints the console line.

**`main()`, the command**:
```
uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855
```
- `--from` copies only `script.json` into a new run folder, so the agent starts from nothing.
- `--run` works in an existing folder.
- `--max-steps`, `--max-tokens`: the budgets.
- `--fail-first TOOL`: the demo failure.
- **`--allow-paid-media`** is required when `MOCK_MEDIA=0`: an agent could otherwise order six paid
  FLUX images on its own (D11).

---

## 11. The real message history after 3 tool calls

From the Uzbek run with an injected failure (`runs/2026-10-22_180241/agent_messages.json`), shortened.
Messages `[5]`–`[8]` and `[11]`–`[13]` are left out (same shape as their neighbours).
```
[ 0] system:    "You are the producer of a short vertical video. ..."
[ 1] user:      "Script: 'Samarqand Registoni kechqurun' (uz, 6 scenes)
                 Scene 1: Tunli Samarqand sehrini ko'rganmisiz? ...  (6 scenes)
                 Already made: images for scenes none; voiceovers for scenes none; final.mp4 not yet."
[ 2] assistant: tool_calls = [
                  {id: call_88714, generate_image, '{"scene":1}', extra_content: {google: {thought_signature: "EmAKXg..."}}},
                  {id: call_88715, generate_image, '{"scene":2}'},
                  ... generate_image 3-6, text_to_speech 1-5 ...
                  {id: call_88725, text_to_speech, '{"scene":6}'} ]          ← 12 calls in ONE answer
[ 3] tool (call_88714): {"ok": true, "scene": 1, "path": "scenes/scene_01.png", "reused": false}
[ 4] tool (call_88715): {"ok": true, "scene": 2, "path": "scenes/scene_02.png", "reused": false}
[ 9] tool (call_88720): {"ok": false, "error": "ConnectionError: the service closed the connection without an answer"}   ← injected
[10] tool (call_88721): {"ok": true, "scene": 2, "path": "audio/scene_02.mp3", "duration_s": 5.52, "words": 9}
[14] tool (call_88725): {"ok": true, "scene": 6, "path": "audio/scene_06.mp3", "duration_s": 4.58, "words": 8}
[15] assistant: tool_calls = [{id: call_99240, text_to_speech, '{"scene":1}', thought_signature...}]   ← retries ONLY the failed one
[16] tool (call_99240): {"ok": true, "scene": 1, "path": "audio/scene_01.mp3", "duration_s": 6.7, "words": 9}
[17] assistant: tool_calls = [{id: call_114883, render_video, '{}', thought_signature...}]
[18] tool (call_114883): {"ok": true, "path": "final.mp4", "duration_s": 28.8, "size_mb": 2.6, "scenes": 6}
[19] assistant: "The video featuring the enchanting night views of Registan Square in Samarkand has been successfully created."
```
What to notice:
- **Four roles only.** Every model turn is one `assistant` message; every tool call gets exactly one
  `tool` message pointing back to it with `tool_call_id`.
- An assistant message that calls tools has **no content** (the model said nothing). The last one has
  content and **no tool calls**, and that's how the loop knows it's done.
- **The error `[9]` is just another tool message**, and `[15]` is the model's reaction to it.
- **What's *not* in the history:** images, audio, visual prompts, ffmpeg output. Only short JSON. The
  model "knows" the video exists only because message `[18]` says so.
- This whole list (plus the tool schemas) is what turn 4 sent: **1,502 prompt tokens**. The API keeps
  nothing between calls: **the history *is* the agent's memory.**

---

## 12. Demo numbers (Gemini 3.5 Flash-Lite, free tier, $0)

### Demo 1: English, no failure (the "done when" run)
| Turn | Prompt tokens | Completion | What the model asked for |
|---|---|---|---|
| 1 | 539 | 180 | **12 parallel calls**: `generate_image` 1–6 + `text_to_speech` 1–6 |
| 2 | 1,313 | 10 | `render_video()` |
| 3 | 1,377 | 21 | nothing: one sentence → **stop: done** |

- **It found the order on its own:** everything independent at once, then the render, then a final answer.
  Nobody told it the steps.
- **Prompt growth:** 539 → 1,313 → 1,377. The +774 is the 12 calls and their 12 results. About **half
  of turn 1's prompt is the tool schemas** (~305 tokens).
- **Same result as the fixed pipeline:** 902 frames (30.07 s), and a **byte-identical `subtitles.srt`**
  compared to Phase 2's `make`. The agent reproduced the pipeline, choosing the steps itself.
- **Cost:** 3,440 tokens, $0.0015 list price, billed $0. **Time:** 371 s, of which **340 s was waiting for
  Gemini** (see E2); our tools took ~31 s.

### Demo 2: Uzbek, the first `text_to_speech` fails on purpose
| Turn | Prompt | Completion | What the model asked for |
|---|---|---|---|
| 1 | 616 | 180 | 12 parallel calls; `text_to_speech(1)` **failed**, the other 11 ok |
| 2 | 1,366 | 16 | `text_to_speech(1)`: **only the failed call** |
| 3 | 1,438 | 10 | `render_video()` |
| 4 | 1,502 | 20 | one sentence → **stop: done** |

- The model **retried exactly the one failed call**, redoing nothing else, and the retry worked.
- It did **not** put `render_video` into the same answer as the retry, because render depends on it,
  and the prompt says to batch only independent calls. So **the failure cost exactly one extra turn**.
- No crash and no traceback, and the run ended normally with `done`.
- Uzbek turn 1 is 616 tokens vs 539 for English: the same 6 scenes split into more tokens (Phase 1, Step 1.2).

### Demo 3: the budgets, live
- `--max-steps 1`: turn 1's 12 calls still ran and were answered, then it **stopped before paying for
  turn 2**. No video, but the history is valid to continue.
- `--max-tokens 2000`: after turn 2 the total was 2,042 ≥ 2,000, so it stopped before turn 3. That's
  **42 tokens over**: the "can overshoot by one turn" rule, in real numbers.
- Both ended normally, with the reason in the log (`"stop": "max_steps"` / `"token_budget"`).

**All of Phase 3:** 13 LLM calls, $0.0055 list price, **billed $0**.

---

## 13. Errors that happened, and what they teach

**E1. Gemini's "thought signatures": proven with a real 400 error.**
- Gemini 3 attaches an encrypted summary of its reasoning to the first tool call of each turn:
  `"extra_content": {"google": {"thought_signature": "EmAKXg..."}}`.
- The build session **tested it on purpose**: the same history sent back *with the signature deleted* was
  rejected:
  ```
  400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts.
  ```
- **The fix is a design rule:** append the assistant message **exactly as returned**
  (`message.model_dump(exclude_none=True)` keeps unknown fields), never rebuild it from the parsed
  `ToolCall`s. A test now fails if anyone "simplifies" this.
- **Lesson:** an "OpenAI-compatible" API is compatible, not identical. Provider-specific details live in
  the fields you didn't write yourself, so pass them through untouched.

**E2. Very slow, very uneven LLM turns (13–254 s): how the cause was found.**
This is a model example of debugging by **testing hypotheses**:
- Hypothesis (a): **hidden SDK retries.** The openai SDK silently retries some errors and can wait up to
  2 minutes (found by reading its source code). *Test:* run with `OPENAI_LOG=info`, which prints a line
  before every retry. Result: **0 retry lines.** Ruled out.
- Hypothesis (b): **the free tier is queueing on Google's side.** *Tests:*
  - a trivial "Reply OK" request (8 tokens) took **21 s**;
  - the **identical** request took **108.5 s, 55.3 s and 45.2 s** on different runs, up to 7× apart for the same input.
- **Conclusion:** the time is spent on Google's side. There's nothing to fix in our code. It also very
  likely explains Phase 2's unexplained 101-second call.
- **Why it was visible at all:** `Turn` logs the LLM's latency separately from each tool's seconds.
  **Measure each part separately, or you can't tell where the time goes.**

**E3. The console stayed empty until the end.** Python **buffers** printed output when it's piped, so
the lines arrived only when the program exited. The log file was fine all along, because it's reopened every
turn. Fix: `$env:PYTHONUNBUFFERED = "1"`.

**E4, E5.** A comment claimed the happy path used ~15k tokens; reality was 3,440, so the comment was fixed.
"after 1 turns" in the summary is now singular. Small, but **comments must match measurements.**

**A suite that never fails proves little.** All 117 tests passed on the first run, so the build session
checked the tests had teeth with a **mutation check**: it deliberately broke the loop in two ways and
confirmed tests fail:
- rebuilding the assistant message (the E1 bug) → 2 tests fail;
- removing the `try/except` → 2 tests fail.

---

## 14. Tests, and your exercise 🧠

- **117 tests** (76 before, +41): test_tools 20, test_loop 20, and +1 for `chat_tools` in test_llm.
- **`ScriptedLLM`:** a fake model that returns a **prepared list of answers**, shaped like Gemini's
  (thought signature included). It lets the tests drive every path of the loop **without any API**:
  - plain answer → done;
  - results go back with the matching `tool_call_id`;
  - the assistant message goes back unchanged;
  - parallel calls all run, in order;
  - a failing tool → `{"ok": false}`, then a retry;
  - an unknown tool → an error result;
  - `max_steps` → exactly N calls;
  - token budget → stop.
- **A whole producer run with the real tools** (voice and ffmpeg faked): the scripted model tries to
  render too early, gets `not ready to render: no image for scenes [1, 2, 3, 4] ...`, fixes it with 8
  parallel calls, renders and finishes.
- **`llm_caller`:** waits 20 s after a fake 429, keeps the 4.5-s pace, and gives up after 4 attempts. No
  real waiting, because `sleep` is faked.

**🧠 Your exercise**, whenever you like, never blocking:
- File: `exercises/phase-3/agent_loop.py`. The data types and `error_result()` are given; **you write
  only the loop** (~35 lines), from the hints in its docstring.
- Check it (terminal, repo root):
  ```
  uv run pytest exercises/phase-3
  ```
  **11 tests**, all red now. They're the real loop tests, so when they're green, you've written an
  agent loop that works the same as the one that made the videos.
- Writing this loop from memory is the best interview preparation in the whole project.

---

## Check questions (optional, answer whenever you like)
**Concepts**
1. Who executes a tool: the model or our code? What exactly does the model produce instead?
2. Why is `arguments` a JSON *string*, and why must our code validate it?
3. What are the four message roles in a tool-using conversation, and what does `tool_call_id` do?
4. Write the agent loop from memory in 8–10 lines of pseudo-code.
5. Why do we need `max_steps` if the model can say "done" by itself? Give two real ways a loop can run forever.
6. A tool fails. Why return the error to the model instead of raising an exception?
7. What do parallel tool calls save, and what must the code do for each call?
8. Why do prompt tokens grow every turn? What does that mean for tool result size?
9. When would you choose a fixed pipeline over an agent?

**Our code**
10. Why do our tools take a scene number instead of the narration text or a file path? Give three reasons.
11. How does `@tool(SceneArgs)` turn a normal function into a tool? Where does the description come from?
12. Why does the loop append `result.message` unchanged instead of rebuilding it? What happened when the signature was removed?
13. Why does `run_loop` receive `call_llm` and `call_tool` as parameters instead of importing them?
14. In demo 2, what did the model do after the failure, and why didn't it render in the same turn?
15. How was the slow-latency cause (E2) found? What two hypotheses were tested, and how?
16. What is a mutation check, and why run one when all tests already pass?

## Interview angles
- "An agent is an LLM, a set of tools and a loop: the model only requests tool calls, my code executes
  them and sends the results back, until the model answers or a budget stops it."
- "Tool errors are returned to the model as results, so it can adapt instead of crashing; hard limits on
  steps and tokens are enforced in code, not trusted to the model."
- "I generate tool schemas from Pydantic models, so the schema the model sees and the validation my code
  runs can't drift apart, and tools take scene numbers, not file paths, for least privilege."
- "I use a workflow when the steps are known and an agent when they aren't; I built the loop by hand
  before using LangGraph, so I know what the framework actually adds."
- "With Gemini over the OpenAI-compatible API, the assistant message must go back exactly as returned:
  it carries a thought signature, and I proved with a real 400 that stripping it breaks tool calling."
- "Given only goals and tools, the agent found the right order itself: 12 parallel calls, then render,
  then done; with an injected failure it retried exactly the failed call and finished one turn later."
- "When the free tier was slow, I didn't guess: I ruled out SDK retries with logging and proved
  server-side queueing by timing identical requests, which is why each turn logs LLM and tool time separately."
