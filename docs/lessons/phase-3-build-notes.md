# Phase 3 build notes (Steps 3.1–3.4: tool use from scratch)

Written by the build session for the teaching session. Everything below happened on this
machine (Windows 11, Python 3.12, uv 0.13.1, openai SDK 3.33.0, pydantic 2.14.1, ffmpeg 9.0.2)
on 2026-10-22. Money spent: **$0** (Gemini free tier, placeholder images, edge-tts, local ffmpeg).
No agent framework: the loop is ~40 lines of plain Python.

**Done-check, in one paragraph.** Given an existing `script.json`, the Producer agent
(`python -m shortsmith.agents.loop --from <run>`) chose its own tool calls and produced
`final.mp4` in 3 turns (English: 12 parallel calls → render → "done"; 3,440 tokens, $0.0015
list price). With the first voice call failing on purpose, the model read the error, retried only
that scene and finished in 4 turns (Uzbek; 5,148 tokens). `--max-steps` and `--max-tokens` both
stopped live runs cleanly. Every turn is in `runs/<id>/agent_log.jsonl`. 117 tests pass; the
exercise fails with `NotImplementedError` as intended. Gemini 3 needs its "thought signatures"
sent back (proven with a real 400), and the free tier was slow (13–254 s per call, server-side).

---

## 1. What was built

### Files

| File | Status | What it is |
|---|---|---|
| `src/shortsmith/agents/__init__.py` | new | package marker |
| `src/shortsmith/agents/tools.py` | new | 3.1: `ToolContext`, the tool registry, the 3 tools, schema generation, `call_tool()` |
| `src/shortsmith/agents/loop.py` | new | 3.2–3.4: `run_loop()`, `error_result()`, `Turn`/`CallLog`/`AgentResult`, `llm_caller()` (pacing + 429 backoff), `journal()` (agent_log.jsonl), the Producer prompt and the CLI `main()` |
| `src/shortsmith/llm.py` | changed | `Message`, `ToolCall`, `ChatResult.tool_calls` + `ChatResult.message`, `chat_tools()` |
| `src/shortsmith/media/tts.py` | changed | `save_voiceovers()` split out of `voice_script()`, so the voice tool can save scene by scene |
| `tests/test_tools.py` | new | 20 tests: schemas, argument validation, each tool in a temporary run folder |
| `tests/test_loop.py` | new | 20 tests: the loop with a scripted fake LLM, error text, the log file, pacing/429, a whole producer run with the real tools |
| `tests/test_llm.py` | changed | the fakes now return the SDK's real `ChatCompletionMessage` (the code calls its `model_dump()`); +1 test for `chat_tools()` |
| `exercises/phase-3/agent_loop.py` + `test_agent_loop.py` | new | the 🧠 exercise: `run_loop()` signature + hints, body `raise NotImplementedError`; 11 tests |

Nothing new in `pyproject.toml` / `uv.lock`: no new dependency (no agent framework, on purpose).

### `llm.py`: what changed for tool calling

- **`Message = dict[str, Any]`**: one chat message in the API's own wire format. Deliberately a dict,
  not a Pydantic model: an assistant message with tool calls nests objects, and Gemini adds fields
  of its own (`extra_content`) that must go back unchanged. A typed model would drop them.
- **`ToolCall(id, name, arguments)`**: one call the model asked for. `arguments` stays a JSON
  *string*: it's the model's output, validated later by the tool's own args model, never trusted.
- **`ChatResult`** got two fields: `tool_calls: list[ToolCall]` (parsed, for our code) and
  `message: Message` (the assistant message exactly as returned, for the history; `repr=False` so it
  doesn't flood prints).
- **`_complete()`** (still the only place that sends a request, so every call is priced and in
  the ledger) now builds both from `response.choices[0].message`:
  `message.model_dump(exclude_none=True)` keeps the SDK's unknown fields; `tool_calls` keeps only
  `type == "function"` calls (the SDK also has "custom" tool calls, which we never send).
- **`chat_tools(messages, tools, model, step="agent", ...)`**: `_complete(..., tools=tools)`. It runs
  nothing: it returns what the model asked for. `tool_choice` is left at the default (`"auto"`: the
  model may answer in words or call tools).

### `agents/tools.py`, piece by piece

- **`ToolContext`** (Pydantic): `run_dir`, `script`, `voiceovers: dict[int, Voiceover]` (filled by
  `text_to_speech`), and the demo switch `fail_first` / `failed`. The model never sees it: it's
  how the *program* binds one run folder to every call. `for_run(run_dir)` loads `script.json`
  (with `Script.load`, so old scripts still validate) and picks up a complete saved
  `voiceovers.json`. `scene(idx)` raises `ValueError("no scene 9: this script has scenes 1-6")`.
- **`Tool`** (Pydantic): `name`, `description`, `args: type[BaseModel]`, `fn`.
  **`TOOLS: dict[str, Tool]`** is the registry; **`@tool(ArgsModel)`** fills it from the function's
  name and docstring.
- **Argument models**: `SceneArgs` (`scene: int = Field(ge=1, description=...)`, `extra="forbid"`)
  for `generate_image` and `text_to_speech`; `NoArgs` for `render_video`.
- **Result models**: `ImageResult(scene, path, reused)`, `VoiceResult(scene, path, duration_s, words)`,
  `VideoResult(path, duration_s, size_mb, scenes)`. Paths are relative to the run folder.
- **`generate_image(ctx, args)`**: `scenes/scene_NN.png` from the scene's `visual_prompt` via
  `images.generate_image(..., step="agent")` (placeholder while MOCK_MEDIA=1, FLUX + ledger line
  otherwise). If the file exists it returns `reused: true` and makes nothing.
- **`text_to_speech(ctx, args)`**: `tts.synthesize(narration, VOICES[language], audio/scene_NN.mp3)`
  (edge-tts, with its own 3 attempts on `NoAudioReceived`). Stores the `Voiceover` in the context;
  once every scene has one, writes `voiceovers.json` (so `make --run` can re-edit the run without voicing).
- **`render_video(ctx, args)`**: first checks every scene has an image file and a voiceover; if
  not, raises `FileNotFoundError("not ready to render: no image for scenes [2, 3]; no voiceover for
  scenes [1]")`, which the model reads and can act on. Then `ffmpeg.assemble()` (Phase 2, unchanged)
  and `ffmpeg.probe()` for the result's duration and size.
- **`_without_titles(schema)`**, **`schema(tool)`**, **`schemas()`**: see §2 and D3.
- **`call_tool(ctx, name, arguments) -> str`**: unknown name → `ValueError("unknown tool 'x'; the
  tools are: generate_image, text_to_speech, render_video")`; `args = tool.args.model_validate_json(arguments
  or "{}")` (invalid JSON, wrong types, missing or extra fields → `ValidationError`); injected
  failure if asked; run; `json.dumps({"ok": True, **result.model_dump(mode="json")}, ensure_ascii=False)`.

### `agents/loop.py`, piece by piece

- **Types**: `StopReason = Literal["done", "max_steps", "token_budget"]`;
  `CallLLM = Callable[[messages, tool_schemas], ChatResult]`; `CallTool = Callable[[name, arguments], str]`.
  The loop gets the LLM and the tools as *functions*: the real ones in `main()`, scripted fakes in the tests.
- **`CallLog`** (id, name, arguments, result, ok, seconds) and **`Turn`** (ts, turn, text,
  tool_calls, prompt_tokens, completion_tokens, tokens_so_far, cost_usd, latency_s, stop) = one
  line of `agent_log.jsonl`. **`AgentResult`** (stop_reason, text, turns, token sums, cost, messages).
- **`error_result(e)`**: an exception → `{"ok": false, "error": "..."}`, ≤ 300 characters (D5).
- **`run_loop(messages, tool_schemas, call_llm, call_tool, max_steps=12, max_tokens=60_000, log=None)`**:
  the loop itself (the exercise):
  ```python
  while True:
      result = call_llm(messages, tool_schemas)
      used += result.total_tokens
      messages.append(result.message)                    # unchanged (thought signatures)
      for call in result.tool_calls:                     # parallel calls: run all, answer each
          try:
              content, ok = call_tool(call.name, call.arguments), True
          except Exception as e:                         # errors are feedback, not crashes
              content, ok = error_result(e), False
          messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
      stop = "done" if no tool calls else "token_budget" if used >= max_tokens
             else "max_steps" if this was turn max_steps else None
      log(Turn(...)); if stop: return AgentResult(...)
  ```
- **`llm_caller(model, ledger_path, min_interval_s=4.5, sleep=time.sleep)`**: returns a `CallLLM`
  that waits so requests start ≥ 4.5 s apart (free tier ~15 requests/minute) and catches
  `openai.RateLimitError` (HTTP 429, Gemini's `RESOURCE_EXHAUSTED`): waits 20, 40, 60 s, gives up
  on the 4th. `sleep` is a parameter so the tests don't really wait.
- **`PRODUCER_SYSTEM_PROMPT`** (verbatim):
  ```
  You are the producer of a short vertical video. The script is already written; your job is to turn it into final.mp4 using your tools.

  - Every scene needs an image and a voiceover before the video can be rendered.
  - You may ask for several tool calls at once when they don't depend on each other.
  - Tools answer with JSON. If one returns "ok": false, read the error and decide: retry, fix the cause, or stop.
  - When final.mp4 is rendered (or it clearly can't be), reply with one short sentence and call no tool.
  ```
  It describes the job, the one dependency, and when to stop; it does not list the steps. The
  tool descriptions carry the details.
- **`task_message(ctx)`**: the user message (an example is in §6, Q3).
- **`journal(path)`**: the `log` callback: appends the `Turn` as one JSON line (file reopened per
  turn, so a crash keeps what happened) and prints the console line.
- **`main()`**: `--run DIR` or `--from DIR` (copies only `script.json` into a new `runs/<timestamp>/`),
  `--max-steps`, `--max-tokens`, `--fail-first TOOL` (repeatable), `--model`, `--allow-paid-media`.
  Writes `agent_log.jsonl` (per turn) and `agent_messages.json` (the whole history at the end),
  prints a summary with the ledger difference.

---

## 2. The tools as the API sees them, and what Gemini sends back

### The three schemas (exactly what `tools.schemas()` sends with every request)

```json
[
  {"type": "function", "function": {
    "name": "generate_image",
    "description": "Make the vertical 9:16 image for one scene, from the script's visual prompt.\nSaves scenes/scene_NN.png. An image that already exists is reused, not made again.",
    "parameters": {"additionalProperties": false,
                   "properties": {"scene": {"description": "Scene number (idx) from the script, starting at 1.", "minimum": 1, "type": "integer"}},
                   "required": ["scene"], "type": "object"}}},
  {"type": "function", "function": {
    "name": "text_to_speech",
    "description": "Record the voiceover for one scene, from the script's narration.\nSaves audio/scene_NN.mp3 plus its word timings (used for the subtitles).",
    "parameters": {"additionalProperties": false,
                   "properties": {"scene": {"description": "Scene number (idx) from the script, starting at 1.", "minimum": 1, "type": "integer"}},
                   "required": ["scene"], "type": "object"}}},
  {"type": "function", "function": {
    "name": "render_video",
    "description": "Edit everything into final.mp4: camera motion, voice, music and subtitles.\nEvery scene needs its image and its voiceover first.",
    "parameters": {"additionalProperties": false, "properties": {}, "type": "object"}}}
]
```

Where each part comes from:
- `name` = the Python function's name; `description` = its docstring (`inspect.getdoc`).
- `parameters` = `SceneArgs.model_json_schema()` / `NoArgs.model_json_schema()` with the `"title"`
  keys removed. `Field(ge=1, description=...)` → `"minimum": 1` + `"description"`;
  `ConfigDict(extra="forbid")` → `"additionalProperties": false`; no default → `"required"`.
- Size: 1,184 characters, ~305 tokens (tiktoken o200k; Gemini's own count is close). They are part
  of the prompt on **every** turn.

### The probe: 3 requests before building the loop (`step="probe"` in the ledger)

A scratch script sent the real schemas with "The script has 2 scenes. Make the image for both
scenes now." Results:

**Request 1** (287 prompt + 28 completion tokens, 13.0 s). The raw assistant message
(`message.model_dump(exclude_none=True)`, signature shortened):

```json
{"role": "assistant",
 "tool_calls": [
   {"id": "call_80370", "type": "function",
    "function": {"name": "generate_image", "arguments": "{\"scene\":1}"},
    "extra_content": {"google": {"thought_signature": "EmAKXgFpFH0TvGkq6w5ft6EI...(132 chars)"}}},
   {"id": "call_80371", "type": "function",
    "function": {"name": "generate_image", "arguments": "{\"scene\":2}"}}
 ]}
```
Facts from this:
- Two tool calls in one answer = **parallel tool calls**, unprompted.
- No `content` at all (it was `null`, and `exclude_none` dropped it): the model said nothing, it only asked.
- `arguments` is a **JSON string**, not an object: our code parses and validates it.
- Only the **first** call of a parallel set carries `extra_content.google.thought_signature`
  (an opaque, encrypted summary of the model's reasoning, 132 characters here).
- Gemini's ids look like `call_80370`; each later tool message must quote its id.
- `render_video`'s empty-object schema was accepted (some older Gemini versions rejected empty `properties`).

**Request 2**: the history with that assistant message *as returned* + two tool messages
(`{"role": "tool", "tool_call_id": "call_80370", "content": "{\"ok\": true, ...}"}`, no `name`
field needed) → 200 OK, 370 prompt + 9 completion tokens, answer
`"I have generated the images for both scenes."`. Even this plain-text answer carried a
`thought_signature` (on the message itself this time).

**Request 3**: the same history, but with `extra_content` deleted from the tool calls →
**rejected**:

```
openai.BadRequestError: Error code: 400 - [{'error': {'code': 400, 'message': 'Function call is
missing a thought_signature in functionCall parts. This is required for tools to work correctly,
and missing thought_signature may lead to degraded model performance. Additional data, function
call `default_api:generate_image` , position 2. Please refer to
https://ai.google.dev/gemini-api/docs/thought-signatures for more details.', 'status': 'INVALID_ARGUMENT'}}]
```

So the brief's warning was right, and this is why `run_loop()` appends `result.message`
unchanged instead of rebuilding it from the parsed `ToolCall` objects (which only keep id, name,
arguments). The openai SDK's models allow extra fields, so `model_dump()` keeps `extra_content`.
(Google's OpenAI-compatibility page only says Gemini 3 "supports OpenAI compatibility for thought
signatures"; the field name and the 400 come from this probe.)

---

## 3. The live demos (Gemini 3.5 Flash-Lite, free tier, MOCK_MEDIA=1)

### Demo 1: English script, no injected failure (the "done when" run)

```
uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855
```
`--from` created `runs/2026-10-22_175609/` and copied only `script.json` into it, so the agent
started from nothing. Console output (verbatim):

```
runs\2026-10-22_175609 | 'Cozy Autumn Coffee in Tashkent' | tools: generate_image, text_to_speech, render_video
turn  1 |   539 in  180 out | 108.5 s | generate_image(1), generate_image(2), generate_image(3), generate_image(4), generate_image(5), generate_image(6), text_to_speech(1), text_to_speech(2), text_to_speech(3), text_to_speech(4), text_to_speech(5), text_to_speech(6)
turn  2 |  1313 in   10 out | 189.8 s | render_video()
turn  3 |  1377 in   21 out |  41.5 s | answer: "The video 'Cozy Autumn Coffee in Tashkent' has been successfully rendered to final.mp4!" | STOP: done

stopped: done after 3 turns | 3229 in + 211 out tokens | ledger $0.001496 list price | 371 s
runs\2026-10-22_175609\final.mp4 | log: runs\2026-10-22_175609\agent_log.jsonl
```

| turn | prompt tokens | completion | tokens so far | LLM time | what the model asked for | tool time |
|---|---|---|---|---|---|---|
| 1 | 539 | 180 | 719 | 108.5 s | **12 parallel calls**: `generate_image` 1–6, `text_to_speech` 1–6 | images 0.28 s in total, voice 9.2 s (1.45–1.67 s each) |
| 2 | 1,313 | 10 | 2,042 | 189.8 s | `render_video()` | 21.4 s |
| 3 | 1,377 | 21 | 3,440 | 41.5 s | nothing: a one-sentence answer → **stop: done** | — |

- **Order the model chose**: everything that doesn't depend on anything else at once (all images
  and all voices in one answer), then the render once all 12 results were `"ok": true`, then a
  final answer. It was never told the order; the system prompt only says "every scene needs an
  image and a voiceover before the video can be rendered" and "you may ask for several tool calls
  at once when they don't depend on each other".
- **Prompt growth** (the history is re-sent every turn): 539 → 1,313 (+774 = the assistant message
  with 12 tool calls + 12 tool results of 72–87 characters each) → 1,377 (+64 = the render call and
  its result). Turn 1's 539 = system prompt + user message + the 3 tool schemas (~305 tokens by
  tiktoken's count, so roughly half of the first prompt is schemas).
- **Completion** 180 tokens for 12 calls ≈ 15 per call. (Gemini 3 bills thinking as output; whether
  this field includes thinking tokens wasn't checked: `usage` has no breakdown here.)
- **Cost**: 3 ledger lines, `step="agent"`, $0.0006117 + $0.0004189 + $0.0004656 = **$0.001496**
  list price; billed $0 (free tier). Images were placeholders: $0.
- **Wall time 371 s**, of which **339.8 s was waiting for Gemini** (see E2: the slow calls) and
  ~31 s was our tools. Phase 2's fixed `make` pipeline needed 47–79 s for the same steps (edge-tts
  was slower that day). What the agent adds is LLM turns, not tool time.
- **Result**: `final.mp4` 1080x1920, 30 fps, **902 frames = 30.07 s**, 2.83 MB, h264/yuv420p + aac.
  Same as Phase 2's `make` output for this script: same 902 frames, **byte-identical
  `subtitles.srt`**, file size within 0.7 KB. The agent reproduced the fixed pipeline, choosing the
  steps itself.
- **Parallel calls: yes**, 12 in one answer. Only the first of the 12 carried a thought signature.

### Demo 2: Uzbek script, the first `text_to_speech` call fails on purpose (Step 3.3 live)

```
$env:OPENAI_LOG = "info"     # the SDK prints a line before every hidden retry (see E2)
uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112757 --fail-first text_to_speech
```
Console output (verbatim, run folder `runs/2026-10-22_180241/`):

```
runs\2026-10-22_180241 | 'Samarqand Registoni kechqurun' | tools: generate_image, text_to_speech, render_video | first call fails on purpose: text_to_speech
turn  1 |   616 in  180 out | 253.6 s | generate_image(1), generate_image(2), generate_image(3), generate_image(4), generate_image(5), generate_image(6), text_to_speech(1) FAILED, text_to_speech(2), text_to_speech(3), text_to_speech(4), text_to_speech(5), text_to_speech(6)
turn  2 |  1366 in   16 out |  82.4 s | text_to_speech(1)
turn  3 |  1438 in   10 out |  82.4 s | render_video()
turn  4 |  1502 in   20 out |  44.0 s | answer: 'The video featuring the enchanting night views of Registan Square in Samarkand has been successfully created.' | STOP: done

stopped: done after 4 turns | 4922 in + 226 out tokens | ledger $0.002046 list price | 492 s
runs\2026-10-22_180241\final.mp4 | log: runs\2026-10-22_180241\agent_log.jsonl
```

| turn | prompt tokens | completion | tokens so far | LLM time | what the model asked for | tool time |
|---|---|---|---|---|---|---|
| 1 | 616 | 180 | 796 | 253.6 s | 12 parallel calls; `text_to_speech(1)` **failed** (injected), the other 11 ok | voice 7.4 s |
| 2 | 1,366 | 16 | 2,178 | 82.4 s | `text_to_speech(1)`: **only the failed call** | 1.6 s |
| 3 | 1,438 | 10 | 3,626 | 82.4 s | `render_video()` | 20.5 s |
| 4 | 1,502 | 20 | 5,148 | 44.0 s | nothing: a one-sentence answer → **stop: done** | — |

How the model reacted to the failure:
- The failure reached it as an ordinary tool message, in the same turn as the 11 successes:
  `{"role": "tool", "tool_call_id": "call_88720", "content": "{\"ok\": false, \"error\": \"ConnectionError: the service closed the connection without an answer\"}"}`
- Turn 2: it retried **exactly** that call, `text_to_speech` for scene 1, with no text and nothing
  else: no images redone, no other voices redone. The retry succeeded (`"duration_s": 6.7, "words": 9`).
- It did **not** put `render_video` into the same answer as the retry: render depends on the
  retry's result, and the system prompt says to batch only calls that don't depend on each other.
  So the failure cost one extra LLM call (the retry turn: 1,366 + 16 tokens, 82 s): 4 turns instead of 3.
- The loop didn't crash, didn't print a traceback, and the run ended with stop reason `done`.

Other numbers:
- Uzbek turn 1 is 616 prompt tokens vs 539 for English: the Uzbek narration (apostrophes, long
  agglutinative words) splits into more tokens for the same 6 scenes.
- `OPENAI_LOG=info` printed **0** retry lines in the whole run (E2).
- Cost $0.0006348 + $0.0004498 + $0.0004564 + $0.0005006 = **$0.002046** list price; billed $0.
- Wall time 492 s: LLM 462.3 s, tools ~30 s.
- `final.mp4`: **863 frames = 28.77 s**, 2.57 MB, 1080x1920. Same 863 frames and byte-identical
  `subtitles.srt` as Phase 2's `make` output for this script (file size within 0.2 KB).

### Demo 3: the two budget stops, live (Step 3.4)

Two short runs on the English script, each in a new run folder (console verbatim; the second
line of each run is the same banner as above and is left out):

```
uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855 --max-steps 1
turn  1 |   539 in  180 out |  55.3 s | generate_image(1), ..., text_to_speech(6) | STOP: max_steps
stopped: max_steps after 1 turns | 539 in + 180 out tokens | ledger $0.000612 list price | 65 s
no final.mp4 | log: runs\2026-10-22_181125\agent_log.jsonl

uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855 --max-tokens 2000
turn  1 |   539 in  180 out |  45.2 s | generate_image(1), ..., text_to_speech(6)
turn  2 |  1313 in   10 out |  25.7 s | render_video() | STOP: token_budget
stopped: token_budget after 2 turns | 1852 in + 190 out tokens | ledger $0.001031 list price | 100 s
runs\2026-10-22_181231\final.mp4 | log: runs\2026-10-22_181231\agent_log.jsonl
```
(The "after 1 turns" typo in the summary line was fixed after this run.)

- **max_steps = 1**: the model's 12 calls in turn 1 were still run and answered (images and voice
  exist in `runs/2026-10-22_181125/`), then the loop stopped *before* paying for turn 2. No video.
  The saved history (`agent_messages.json`, 15 messages) ends with 12 tool messages, so it could be
  continued later by sending it again.
- **max_tokens = 2000**: after turn 1, 719 tokens were used (< 2000), so turn 2 happened; turn 2
  pushed the total to 2,042 ≥ 2000, its `render_video` call had already run (so `final.mp4`
  exists), and the loop stopped before turn 3, the model's closing sentence. This is the "checked
  after each turn, can overshoot by one turn" rule (D9) in real numbers: 42 tokens over.
- Both stops end the run normally, with the reason on the last log line (`"stop": "max_steps"` /
  `"stop": "token_budget"`) and in the summary. Nothing is raised.
- These three Gemini calls took 25.7–55.3 s each: much faster than an hour earlier (E2).

### All Phase 3 LLM calls (from `runs/ledger.jsonl`)

| time (UTC) | step | prompt | completion | list price |
|---|---|---|---|---|
| 12:54:28 | probe 1 (tools) | 287 | 28 | $0.0001561 |
| 12:54:48 | probe 2 (history as returned) | 370 | 9 | $0.0001335 |
| — | probe 3 (signature stripped) | rejected with 400, not billed, so not in the ledger | | |
| 12:57:57 – 13:02:19 | demo 1, 3 turns | 3,229 | 211 | $0.0014962 |
| 13:06:54 – 13:10:53 | demo 2, 4 turns | 4,922 | 226 | $0.0020416 |
| 13:07:03 | latency probe ("reply OK") | 8 | 1 | $0.0000049 |
| 13:12:20 | demo 3a (`--max-steps 1`) | 539 | 180 | $0.0006117 |
| 13:13:17 – 13:13:52 | demo 3b (`--max-tokens 2000`) | 1,852 | 190 | $0.0010306 |
| **total** | **13 calls** | | | **$0.0054746** (billed $0) |

Ledger total after the phase: $0.0189463 list price.

---

## 4. Errors hit, how they were found, how they were fixed

**E1. Gemini rejects a history without its thought signature (provoked on purpose).** Probe request 3
(§2): `400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts`.
Found by deleting `extra_content` from the returned assistant message before sending it back.
"Fix" = the design: append `result.message` exactly as returned (D7), and a test that fails if the
loop rebuilds it (see the mutation check in §7).

**E2. Very slow and very uneven LLM turns: 13–254 s per call.** Demo 1's three calls took 108.5 s,
189.8 s and 41.5 s; the probe's calls a few minutes earlier took 13 s each; Phase 2 had one unexplained 101-s
writer call (its E9). Two hypotheses:
- (a) *hidden SDK retries*: the openai SDK retries 429/5xx twice by default and honours a server
  `retry-after` header up to `MAX_RETRY_AFTER_DELAY = 2 * 60` s **silently** (found by reading
  `openai/_constants.py` and `_base_client.py`), which could add up to minutes;
- (b) *server-side queueing on the free tier*.
How it was tested:
1. Demo 2 ran with `OPENAI_LOG=info`, which makes the SDK print
   `Retrying request in X seconds (retry n of 2)` before every retry (checked first against a closed
   local port: `Retrying request in 0.476138 seconds (retry 1 of 1)` appears). Result: 4 calls of
   253.6, 82.4, 82.4 and 44.0 s, and **0 retry lines**. So (a) is ruled out.
2. While demo 2's first call was pending, a trivial request ("Reply with the single word OK.")
   took **21.1 s**, with `usage = {'completion_tokens': 1, 'prompt_tokens': 8, 'total_tokens': 9}`:
   the endpoint itself was slow, and there were no hidden thinking tokens to explain the time.
3. The same request, sent again, took very different times. Turn 1 of demos 1, 3a and 3b was the
   identical request (539 prompt + 180 completion tokens each time): **108.5 s, 55.3 s, 45.2 s**.
   Turn 2 of demos 1 and 3b (1,313 + 10): **189.8 s vs 25.7 s**.
Conclusion: (b). Same input, up to 7x different latency means the time is spent on Google's side
(free-tier queueing/load), not in our code, not in SDK retries, not in the prompt size. Nothing to
fix in the code. That's why `Turn` logs the LLM's `latency_s` separately from each tool's `seconds`,
which is how this was visible at all. It very likely explains Phase 2's E9 (one 101-s writer call)
too. If it ever matters: a timeout shorter than the SDK's 600 s default plus a retry, or a paid key.

**E3. The console capture of demo 1 was empty until the very end (tooling).** Python buffers stdout
when it's piped (`... | Tee-Object`), so the per-turn lines appeared only when the process exited.
`agent_log.jsonl` was readable all along because `journal()` reopens the file for every turn.
Fix for later runs: `$env:PYTHONUNBUFFERED = "1"`. (PowerShell's `Tee-Object` also writes a BOM
(U+FEFF) at the start of the file: cosmetic.)

**E4. A wrong number in a comment.** Before any run, `MAX_TOKENS` was commented "the happy path used
~15k". The real happy path used 3,440 tokens (demo 1). Fixed the comment after the demo; the
default 60,000 stays (room for many retries; a stuck model still can't run up a bill).

**E5. "after 1 turns".** The summary line had no singular (seen in demo 3a). Fixed.

No test failed during the build: all 117 passed on their first run. Because a suite that never
failed proves little, its teeth were checked on purpose (§7, mutation check).

---

## 5. Design decisions and why

**D1. Tools take scene numbers; the content comes from the script bound in `ToolContext`.**
`generate_image(scene=2)` reads scene 2's `visual_prompt` itself; `text_to_speech(scene=2)` reads its
narration. Three reasons: (a) the script stays the single source of truth (the model can't
"improve" an Uzbek narration and break the apostrophes or the subtitles); (b) arguments are a few
tokens (`{"scene":2}` = 5 characters) instead of a 30-word prompt the model would have to copy,
and every argument is paid twice: as output now, as input on every later turn; (c) no tool takes a
path, so nothing the model writes can point outside the run folder. Model text is never a command:
it can only choose among three functions and one integer.

**D2. One Pydantic model per tool's arguments → the JSON schema is generated from it.**
Same idea as `chat_json`'s `response_format`: `SceneArgs.model_json_schema()` is what the API sees,
and `SceneArgs.model_validate_json(arguments)` is what checks the model's answer, so they can't
disagree. `Field(ge=1, description=...)` becomes `"minimum": 1` and a description in the schema;
`extra="forbid"` becomes `"additionalProperties": false` and makes an invented argument an error.
`render_video` takes `NoArgs` (an empty object schema; Gemini accepted it, see §2).

**D3. Pydantic's automatic `"title"` keys are stripped from the schemas.** They repeat the field and
class names ("Scene", "SceneArgs") and tell the model nothing. Tool schemas are re-sent with every
request: measured with tiktoken (o200k), 338 → 305 tokens for the three tools, i.e. 33 tokens on
every turn. `_without_titles()` keeps a *property* that happens to be called `title` (tested).

**D4. A tool's docstring is its description.** The `@tool(ArgsModel)` decorator registers
name = function name, description = docstring, args model, function. FastMCP's `@mcp.tool()` does
exactly this in Phase 4, so moving the tools means re-registering the same functions. These
docstrings are written *for the model* (what the tool does, what it saves, what it needs first),
not as "why" comments.

**D5. Results are short JSON with paths relative to the run folder.**
`{"ok": true, "scene": 2, "path": "scenes/scene_02.png", "reused": false}` = 72 characters. Every
result stays in the history and is paid for again on every later turn. Errors are capped at 300
characters; a `ValidationError` becomes `invalid arguments: scene: Input should be a valid
integer, ...` (no pydantic URL); an ffmpeg `CalledProcessError` becomes `ffmpeg failed with exit
code 1 | ffmpeg said: ...` (its `str()` would quote the whole ~1,500-character command).

**D6. The registry raises; the loop converts every exception into an error result.**
`call_tool()` raises on an unknown name, bad arguments or a failing tool; `run_loop()` catches
`Exception` around each call and sends `error_result(e)` back as that call's tool message. One
`try/except` in one place guarantees that *no* tool, including future MCP tools, can crash the
loop, and the registry stays a plain table of functions. (`KeyboardInterrupt` is not an
`Exception`, so Ctrl+C still stops a run.)

**D7. The assistant message is appended exactly as the API returned it.** `ChatResult.message` is
`message.model_dump(exclude_none=True)`; the SDK's models keep unknown fields, so Gemini's
`extra_content.google.thought_signature` survives. Rebuilding the message from `ToolCall` objects
gets a 400 (proven in §2). `exclude_none=True` drops `"content": null` instead of sending it back.

**D8. Parallel tool calls run one after another, in the order the model gave them.**
Deterministic and simple: `render_video` placed after the voice calls in the same answer would
still find them done. Each call gets its own tool message with its own `tool_call_id`, in order.
(Running the 6 `text_to_speech` calls concurrently would save ~8 s; not worth the complexity yet.)

**D9. Stop rules, checked after each turn in this order: done → token budget → max steps.**
- *done*: the model answered without a tool call. This wins even over budget: a finished job is finished.
- *token_budget*: `tokens_so_far >= max_tokens` (prompt + completion of all turns). Checked before
  paying for another call, so it can overshoot by at most one turn.
- *max_steps*: this was LLM turn number `max_steps`.
The tool calls of the last turn are still executed and answered before stopping, so the saved
history is always valid to continue (every tool call has its tool message). Defaults: 12 turns,
60,000 tokens (see §3 for what a real run used).

**D10. Pacing and 429 handling live in `llm_caller()`, not in the loop.** The loop only knows
"call the LLM"; the free-tier rules (one request per 4.5 s, wait 20/40/60 s after a 429, give up
after 4 attempts) belong to the provider. The scripted fake LLM in the tests needs none of it. A
429 is not billed, so only the attempt that succeeds lands in the ledger.

**D11. Paid media needs an explicit flag.** With `MOCK_MEDIA=0` the agent could order 6 FLUX images
on its own, which CLAUDE.md forbids without asking. `main()` refuses unless `--allow-paid-media`
is given. Images that already exist are reused (`"reused": true`), so a retrying model never pays
twice for a scene.

**D12. The injected failure is realistic and honest in the log.** `--fail-first TOOL` makes that
tool's first *valid* call raise `ConnectionError: the service closed the connection without an
answer` (arguments are validated first, so a bad call doesn't use up the injection). The model
sees an ordinary service error; agent_log.jsonl records the failed call (`"ok": false`), and the
console banner says the failure is on purpose.

**D13. Two files per run.** `agent_log.jsonl`: one line per turn, appended as it happens (a crash
keeps the turns so far), short enough to read. `agent_messages.json`: the whole conversation at the
end, exactly what was sent, including thought signatures (they're opaque, not secret).

**D14. The task message is compact and states what already exists.** The user message holds the
title, language and each scene's narration (no visual prompts: the tools read them), plus "Already
made: images for scenes [...]; voiceovers for scenes [...]; final.mp4 ...". A half-finished run
isn't redone; a complete saved voice is loaded into the context so `render_video` can use it.

**D15. The exercise tests are the real loop tests.** `exercises/phase-3/test_agent_loop.py` is the
"loop" section of `tests/test_loop.py`; the data types and `error_result()` are given, so the owner
writes only the loop itself (~35 lines). The same 11 tests pass against the real `run_loop`
(checked in a scratch folder), so the exercise is solvable as written.

---

## 6. The three plan questions, answered from the real runs

**Q1. Who executes the tool, the model or our code?** Our code, always. All the model produced in
demo 1, turn 1 was text in a fixed shape:
`{"id": "call_214025", "type": "function", "function": {"name": "generate_image", "arguments": "{\"scene\":1}"}}`.
Then *our* process looked the name up in `TOOLS`, validated `{"scene":1}` with `SceneArgs`, ran
`generate_image()`, which wrote `scenes/scene_01.png` 0.07 s later, and put
`{"ok": true, "scene": 1, "path": "scenes/scene_01.png", "reused": false}` into a tool message.
The model never saw the image, the file system or ffmpeg; it saw 72 characters of JSON. Our code
could just as well refuse: unknown names, invalid arguments and paid media without
`--allow-paid-media` never run. Demo 2 shows the other side: when *our* code failed (the injected
`ConnectionError`), the model only learned about it because our code wrote the error into the
tool message. "The model calls a tool" really means "the model asks; the program decides and runs".

**Q2. Why do we need `max_steps`?** Because nothing in the API ends the loop: it ends only when
the model *chooses* to answer without a tool call. A model that keeps retrying a tool that can
never succeed (ffmpeg broken, a quota used up), or keeps "re-checking" its work, would call
forever, and every turn costs more than the last because the whole history is re-sent (demo 2:
616 → 1,366 → 1,438 → 1,502 prompt tokens; 5,148 tokens for 4 turns). A retry is useful once
(demo 2), harmful forever. `max_steps` caps the number of LLM calls; the token budget caps the
total cost even when each turn is huge. Live: `--max-steps 1` stopped after the first turn's 12
tool calls with `"stop": "max_steps"` and a history that is still valid to continue; `--max-tokens
2000` stopped after turn 2 at 2,042 tokens (demo 3). In the tests, a scripted model that never
stops is cut off after exactly `max_steps` calls.

**Q3. What is in the message history after 3 tool calls?** Real history of demo 2
(`runs/2026-10-22_180241/agent_messages.json`), after its 3 tool-calling turns plus the final
answer: 20 messages. Thought signatures shortened to 16 characters (they are 132), JSON keys put in reading order; the
system prompt is the one in §1; `[5]`–`[8]` (images 3–6) and `[11]`–`[13]` (voices 3–5) are left out, they
have the same shape as their neighbours.

```
[ 0] {"role": "system", "content": "You are the producer of a short vertical video. ..."}
[ 1] {"role": "user", "content": "Script: 'Samarqand Registoni kechqurun' (uz, 6 scenes)\nScene 1: Tunli Samarqand sehrini ko'rganmisiz? Registon maydoni yulduzlar ostida porlaydi.\nScene 2: Qadimiy madrasalar chiroqlari qorong'i osmon fonida juda jozibali ko'rinadi.\nScene 3: Bu yerda o'tmish nafasi va sharqona ertaklar muhiti hukm surmoqda.\nScene 4: Tun sukunatida qadimgi devorlar go'yo o'tmish sirlarini pichirlayotgandek tuyuladi.\nScene 5: Har bir naqsh va gumbaz o'ziga xos boy tarixni hikoya qiladi.\nScene 6: Ushbu go'zallikni ko'rish uchun hoziroq Samarqandga safar qiling.\n\nAlready made: images for scenes none; voiceovers for scenes none; final.mp4 not yet."}
[ 2] {"role": "assistant", "tool_calls": [
       {"id": "call_88714", "type": "function", "function": {"name": "generate_image", "arguments": "{\"scene\":1}"}, "extra_content": {"google": {"thought_signature": "EmAKXgFpFH0TQIHl..."}}},
       {"id": "call_88715", "type": "function", "function": {"name": "generate_image", "arguments": "{\"scene\":2}"}},
       ... 9 more: generate_image 3-6, text_to_speech 1-5 (call_88716 ... call_88724) ...
       {"id": "call_88725", "type": "function", "function": {"name": "text_to_speech", "arguments": "{\"scene\":6}"}}]}
[ 3] {"role": "tool", "tool_call_id": "call_88714", "content": "{\"ok\": true, \"scene\": 1, \"path\": \"scenes/scene_01.png\", \"reused\": false}"}
[ 4] {"role": "tool", "tool_call_id": "call_88715", "content": "{\"ok\": true, \"scene\": 2, \"path\": \"scenes/scene_02.png\", \"reused\": false}"}
[ 9] {"role": "tool", "tool_call_id": "call_88720", "content": "{\"ok\": false, \"error\": \"ConnectionError: the service closed the connection without an answer\"}"}
[10] {"role": "tool", "tool_call_id": "call_88721", "content": "{\"ok\": true, \"scene\": 2, \"path\": \"audio/scene_02.mp3\", \"duration_s\": 5.52, \"words\": 9}"}
[14] {"role": "tool", "tool_call_id": "call_88725", "content": "{\"ok\": true, \"scene\": 6, \"path\": \"audio/scene_06.mp3\", \"duration_s\": 4.58, \"words\": 8}"}
[15] {"role": "assistant", "tool_calls": [{"id": "call_99240", "type": "function", "function": {"name": "text_to_speech", "arguments": "{\"scene\":1}"}, "extra_content": {"google": {"thought_signature": "EmAKXgFpFH0TPOrg..."}}}]}
[16] {"role": "tool", "tool_call_id": "call_99240", "content": "{\"ok\": true, \"scene\": 1, \"path\": \"audio/scene_01.mp3\", \"duration_s\": 6.7, \"words\": 9}"}
[17] {"role": "assistant", "tool_calls": [{"id": "call_114883", "type": "function", "function": {"name": "render_video", "arguments": "{}"}, "extra_content": {"google": {"thought_signature": "EmAKXgFpFH0TDaP1..."}}}]}
[18] {"role": "tool", "tool_call_id": "call_114883", "content": "{\"ok\": true, \"path\": \"final.mp4\", \"duration_s\": 28.8, \"size_mb\": 2.6, \"scenes\": 6}"}
[19] {"role": "assistant", "content": "The video featuring the enchanting night views of Registan Square in Samarkand has been successfully created.", "extra_content": {"google": {"thought_signature": "EmAKXgFpFH0TKsHt..."}}}
```

What to notice:
- Four roles only: `system`, `user`, `assistant`, `tool`. Every model turn is one `assistant`
  message; every tool call gets exactly one `tool` message that points back to it with `tool_call_id`.
- An assistant message that calls tools has **no `content`** (the model said nothing; `null` was
  dropped by `exclude_none`). The last one has content and no `tool_calls`: that is how the loop knows
  it's done.
- The `thought_signature` sits on the first tool call of each assistant turn (and on the final
  answer). It is opaque; we only carry it.
- The error `[9]` is just another tool message. Message `[15]` is the model's reaction to it.
- What is **not** in the history: images, audio, visual prompts, ffmpeg output, file contents. Only
  short JSON. The model "knows" the video exists only because message `[18]` says so.
- This whole list (system + user + 12 + 12 + 2 + 2 + the tool schemas) is what turn 4 sent:
  1,502 prompt tokens. The API keeps nothing between calls; the history *is* the agent's memory.

---

## 7. Tests

- `uv run pytest`: **117 passed** (~4 s; 76 before this phase, +41). Per file: test_ffmpeg 31,
  test_loop 20, test_tools 20, test_tts 10, test_models 9, test_llm 8, test_ledger 8, test_make 7, test_images 4.
- What the new tests cover:
  - `test_tools.py` (20): schemas match the args models (exact JSON), no `"title"` keys, a property
    called `title` survives, empty-object schema for `render_video`; unknown tool names the real ones;
    5 kinds of bad arguments → `ValidationError` (wrong type, `scene: 0`, missing, extra field, not
    JSON); scene 9 → `ValueError`; `""` means no arguments; images: short JSON result (< 100 chars),
    reuse without paying; voice: `voiceovers.json` only once complete; render: the exact "not ready"
    message, and the happy path with faked ffmpeg; `--fail-first` fails only the first *valid* call;
    a complete saved voice is picked up by `for_run`.
  - `test_loop.py` (20): a scripted fake LLM (`ScriptedLLM`: a list of prepared `ChatResult`s,
    shaped like Gemini's, thought signature included; it deep-copies every request) and
    `FakeTools` drive the loop: plain answer → done; schemas sent every turn; result goes back with
    the matching `tool_call_id`; the assistant message goes back unchanged (signature included);
    parallel calls all run, answered in order; a failing tool → `{"ok": false, ...}` and the model
    retries; unknown tool → error result; `max_steps` → exactly N calls, history still valid;
    token budget → stops at 1,200 ≥ 1,000; a final answer over budget is still "done"; log/totals.
    Then the real-code-only parts: `error_result` for a `ValidationError` (no URL), for ffmpeg's
    `CalledProcessError` (its message, not the command) and length capping; `journal()` writes
    parseable JSON lines and prints `generate_image(1) FAILED`; `llm_caller` waits 20 s after a
    fake 429 and still keeps the 4.5-s pace, and gives up after 4 attempts; a whole producer run
    with the **real** registry + placeholder images (voice and ffmpeg faked) where the scripted model
    renders too early, gets `not ready to render: no image for scenes [1, 2, 3, 4]; no voiceover
    for scenes [1, 2, 3, 4]`, fixes it with 8 parallel calls, renders, finishes; the task message;
    `MOCK_MEDIA=0` without `--allow-paid-media` exits.
  - `test_llm.py` (+1): `chat_tools` sends the schemas, parses `ToolCall`s, keeps `extra_content`,
    drops `content: null`, records the ledger line with `step="agent"`.
- `uv run pytest exercises/phase-3`: **11 failed**, every one with `NotImplementedError: your turn:
  follow the hints above`. The same 11 pass when `agent_loop.py` just re-exports the real
  `run_loop` (scratch folder). `uv run pytest exercises` runs both phases together: 27 failed
  (16 + 11), no file-name clashes.
- **Mutation check** (are the tests sharp?): two broken copies of `run_loop` were run against
  the exercise tests in a scratch folder.
  - Rebuilding the assistant message from `ToolCall`s instead of appending `result.message`:
    2 tests fail, among them
    `test_the_assistant_message_goes_back_unchanged_thought_signature_included`, the bug that would
    give the live 400 from E1.
  - Removing the `try/except` around `call_tool`: 2 tests fail
    (`test_a_failing_tool_is_reported_to_the_model_which_can_retry`, `test_an_unknown_tool_is_reported_to_the_model_too`).

---

## 8. Open items

- **Gemini free-tier latency** (E2): 13–254 s per call today, server-side. Fine for learning; a
  demo video for the README should be recorded at a quiet hour (or the console log shown instead).
- **No real FLUX image went through the agent** ($0 rule). `--allow-paid-media` exists; the first
  paid agent run should be one scene, asked for first.
- **Parallel calls run one by one.** The 6 voice calls take 7–9 s in a row; concurrent execution
  (threads or asyncio) would cut that to ~2 s. Only worth it if voice becomes the bottleneck.
- **The loop has no context trimming.** Fine at 1.5k tokens per turn; Phase 6.9 (context engineering)
  is where old tool results would be summarized or dropped.
- **Phase 4 (MCP)**: the registry (`TOOLS`: name, docstring description, args model, function)
  maps onto FastMCP's `@mcp.tool()`; `ToolContext` would become the server's per-run state,
  and `call_tool()`'s "raise on error" fits MCP's `isError` results. The loop's `call_tool`
  parameter is where an MCP client goes instead of the direct import.
- Phase 2's E9 (one 101-s writer call) is very likely the same free-tier latency as E2.
- Russian word target (Phase 2 open item) is still unmeasured.
