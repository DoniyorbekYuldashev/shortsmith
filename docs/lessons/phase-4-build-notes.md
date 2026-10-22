# Phase 4 build notes (Steps 4.1–4.4: MCP)

Written by the build session for the teaching session. Everything below happened on this
machine (Windows 11, Python 3.12, uv 0.13.1, **mcp 2.3.0** (the official Python SDK), pydantic
2.14.1, ffmpeg 9.0.2) on 2026-10-22. Money spent: **$0** (Gemini free tier, placeholder images,
edge-tts, local ffmpeg).

**Done-check, in one paragraph.** `src/shortsmith/mcp_server.py` is a real MCP server (stdio)
that publishes the three Phase 3 media tools straight from the Phase 3 registry (a `run_id`
added), two tools to start a run (`create_run` takes a whole Script and our Pydantic model
validates it; `write_script` uses our Gemini writer), one resource plus two resource templates,
and one prompt. The official SDK client listed and called it over a real subprocess (initialize,
tools/list, tools/call, an error result, resources/read, prompts/get: all captured on the wire,
§3), and nothing but JSON-RPC ever reached stdout (a test proves it, and found a real leak, E3).
Our Phase 3 agent loop, **unchanged**, made `final.mp4` through the server
(`loop --from runs/2026-10-22_112855 --mcp`): 3 turns, the same 539 / 1,313 / 1,377 prompt tokens
as the direct Phase 3 run, the same 902 frames, byte-identical images, subtitles and all 13 tool
messages (§5). A second live run with an injected failure showed the model reading an MCP error
result and retrying (§5). The exact Claude Desktop launch command was tested from a foreign
folder with a minimal environment; the config snippet and steps for the owner are in §4.
154 tests pass (117 + 37). The owner still has to do the Claude Desktop test himself.

---

## 1. What was built

### Files

| File | Status | What it is |
|---|---|---|
| `pyproject.toml`, `uv.lock` | changed | `uv add "mcp[cli]"` → `mcp>=2.3.0` (+ `mcp-types`, starlette, uvicorn, jsonschema, pywin32, typer, rich, python-dotenv, ...: 25 packages) |
| `src/shortsmith/mcp_server.py` | new | 4.2: the MCP server: tools, resources, prompt, `main()` (stdio) |
| `src/shortsmith/agents/mcp_tools.py` | new | 4.4: the client side: `server_params()`, `chat_schema()`, `tool_message()`, `McpToolbox` |
| `src/shortsmith/agents/tools.py` | changed | `run_tool()` split out of `call_tool()`; `ok_json()`; `describe_error()` + `ToolFailed` + `MAX_ERROR_CHARS` moved here from loop.py; per-scene voice timings (`timing_file()`, `load_voiceover()`); `RunStatus` / `run_status()` |
| `src/shortsmith/agents/loop.py` | changed | `error_result()` uses `describe_error()`; `task_message(script, status)`; `produce()` (the shared part of main), `produce_over_mcp()`, the `--mcp` flag. **`run_loop()` is untouched.** |
| `tests/test_mcp_server.py` | new | 25 tests: the server through the SDK's in-process client, plus 2 real subprocess tests |
| `tests/test_mcp_tools.py` | new | 11 tests: the bridge, the whole loop through MCP, the `--mcp` CLI |
| `tests/test_loop.py` | changed | the 2 `task_message()` call sites |
| `tests/test_tools.py` | changed | +1 test: a half-voiced run's voice comes back from the per-scene files |
| `exercises/phase-4/mcp_bridge.py` + `test_mcp_bridge.py` | new | optional 🧠 exercise: `chat_schema()` and `tool_message()`; 9 tests |

### First surprise: the SDK is version 2, and FastMCP is now called `MCPServer`

`uv add "mcp[cli]"` installed **mcp 2.3.0**. In v2:
- `from mcp.server.fastmcp import FastMCP` (what most tutorials show) raises
  `ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was
  renamed to MCPServer (from mcp.server.mcpserver import MCPServer) and other APIs changed; see
  the migration guide ... or pin 'mcp<2'`. (Found by reading the package before writing code:
  `mcp/server/fastmcp.py` is a stub that only raises this.) Same decorators: `@mcp.tool()`,
  `@mcp.resource(uri)`, `@mcp.prompt()`, `mcp.run("stdio")`.
- The protocol types live in a separate package, `mcp_types`, with snake_case attributes
  (`input_schema`, `is_error`, `structured_content`) and camelCase on the wire (`inputSchema`, `isError`).
- There is a high-level `mcp.Client` that takes a URL (HTTP), a `StdioServerParameters`
  (starts a subprocess), or a server object (in-process, for tests).
- Two protocol versions: **2025-11-25** with the classic `initialize` handshake, and the new
  **2026-07-28**, which drops the handshake (`server/discover` + metadata on every request).
  `Client(mode="auto")` (the default) picks the new one if the server has it; `mode="legacy"`
  forces the handshake. Both captured in §3.
- `stdio_server()` now protects the channel itself: "While serving, fd 0 points at the null
  device and fd 1 at stderr, so handlers and children read EOF and their stray output misses
  the wire" (its docstring). Verified in §2 and E3. v1 did not do this.

### `mcp_server.py`, piece by piece

- **`mcp = MCPServer(name="shortsmith", instructions=...)`**: the `instructions` string goes to
  the host in the initialize answer (hosts may put it into the model's context): "Makes
  ~30-second vertical short videos ... Flow: create_run with a script you write (or write_script)
  -> generate_image and text_to_speech for every scene -> render_video, which saves final.mp4 in
  the run folder."
- **`RUNS = Path("runs")`**, relative like everywhere else in the project; `main()` moves into
  `--root` (default: the repo, found from `__file__`) before serving.
- **`RunId`** = `Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$", description=...)]`: a
  folder name, never a path.
- **`run_dir(run_id)`**: checks the pattern again (resources get the id from a URI, not
  through the schema) and that `runs/<id>/script.json` exists; else
  `ValueError("no run 'nope': start one with create_run or write_script")`.
- **`new_run_dir()`**: `runs/<YYYY-MM-DD_HHMMSS>`, with `-2`, `-3` if two runs start in the same second.
- **`media_tool(t)`**: turns one registry `Tool` into an MCP tool function (D1). The function
  body: load a fresh `ToolContext` from disk for that run, share the server's `FAILED` set, call
  `tools.run_tool()`, and turn any exception into `ToolError(tools.describe_error(e))`. Its
  `__signature__` is built from the registry's args model plus `run_id`, and its return
  annotation is the registry function's (`ImageResult`, ...), so MCPServer generates the input
  schema and an output schema from them.
  `for t in tools.TOOLS.values(): mcp.add_tool(media_tool(t), name=t.name, description=t.description)`.
- **`create_run(script: Script) -> RunStarted`** (`@mcp.tool(description=CREATE_RUN)`): MCPServer
  validates the argument against `Script` before the function runs: hook length, 4–8 scenes,
  idx order, the word-count rule. The function only saves `script.json` in a new run folder.
  `CREATE_RUN` (the description) is built from the same constants the validator uses
  (`SCENES_MIN`, `NARRATION_WORDS`, `HOOK_MAX_WORDS`, ...).
- **`write_script(topic, language="en") -> ScriptWritten`**: `llm.write_script()` (one Gemini
  call, ledger step `"writer"`), then the same run start; returns the run plus the whole script.
  A writer answer that breaks the rules → `ToolError("the writer's script broke the rules, try again: ...")`.
- **Resources**: `shortsmith://runs` (static), `shortsmith://runs/{run_id}/script` and
  `shortsmith://runs/{run_id}/status` (templates). A missing run → `ResourceNotFoundError`, which
  MCPServer sends as a JSON-RPC **protocol error** (code -32602), not as a result.
- **Prompt**: `make_short(topic, language="en")`, title "Make a short video". Returns one user
  message that includes `llm.WRITER_SYSTEM_PROMPT` verbatim (reuse, not a copy) and the 3 steps.
- **`main()`**: `--root`, `--fail-first TOOL` (repeatable, like Phase 3's demo switch, but on the
  server: the loop's `--fail-first` is passed through to it); `os.chdir(root)`; plain one-line
  logging on stderr; `sys.stdout.reconfigure(line_buffering=True)` (E3); one startup log line;
  `mcp.run("stdio")`, which returns when the host closes stdin.

### The tools as an MCP host sees them

The real `tools/list` answer (captured, §3) has 5 tools, 7,052 characters, **2,061 tokens**
(tiktoken o200k). For one media tool, exactly as sent:

```json
{"name": "generate_image",
 "description": "Make the vertical 9:16 image for one scene, from the script's visual prompt.\nSaves scenes/scene_NN.png. An image that already exists is reused, not made again.",
 "inputSchema": {"type": "object", "title": "generate_imageArguments", "required": ["run_id", "scene"],
   "properties": {
     "run_id": {"type": "string", "title": "Run Id", "pattern": "^[A-Za-z0-9_-]{1,64}$",
                "description": "The run's id, as returned by create_run or write_script."},
     "scene": {"type": "integer", "title": "Scene", "minimum": 1,
               "description": "Scene number (idx) from the script, starting at 1."}}},
 "outputSchema": {"type": "object", "title": "ImageResult", "required": ["scene", "path", "reused"],
   "properties": {"scene": {"type": "integer", "title": "Scene"}, "path": {"type": "string", "title": "Path"},
                  "reused": {"type": "boolean", "title": "Reused"}}}}
```

| tool | input (besides the model's own words) | output (structuredContent) | description from |
|---|---|---|---|
| `generate_image` | `run_id`, `scene` (int ≥ 1) | `ImageResult(scene, path, reused)` | Phase 3 docstring |
| `text_to_speech` | `run_id`, `scene` | `VoiceResult(scene, path, duration_s, words)` | Phase 3 docstring |
| `render_video` | `run_id` | `VideoResult(path, duration_s, size_mb, scenes)` | Phase 3 docstring |
| `create_run` | `script` (the whole `Script` schema with `$defs` Script/Scene) | `RunStarted(run_id, folder, title, language, scenes)` | `CREATE_RUN`, built from the validator's constants |
| `write_script` | `topic`, `language` (enum en/uz/ru, default en) | `ScriptWritten` = `RunStarted` + `script` | its docstring |

| resource | kind | content |
|---|---|---|
| `shortsmith://runs` | static (in `resources/list`) | JSON list, newest first: `run_id`, `title`, `language`, `scenes`, `final` |
| `shortsmith://runs/{run_id}/script` | template (in `resources/templates/list`) | the run's `script.json` |
| `shortsmith://runs/{run_id}/status` | template | `{"images": [1], "voiceovers": [], "final": false}` |

| prompt | arguments | returns |
|---|---|---|
| `make_short` ("Make a short video") | `topic` (required), `language` (optional) | one user message: topic, language, the writer's brief, the 3 steps |

### `agents/mcp_tools.py`, piece by piece

- **`server_params(root, fail_first)`**: `StdioServerParameters(command=sys.executable,
  args=["-m", "shortsmith.mcp_server", "--root", root, ...], env=<only PASS_ENV>, cwd=root)`.
  `stdio_client` gives the child only a short list of "safe" variables (on Windows: APPDATA,
  HOMEDRIVE, HOMEPATH, LOCALAPPDATA, PATH, PATHEXT, PROCESSOR_ARCHITECTURE, SYSTEMDRIVE,
  SYSTEMROOT, TEMP, USERNAME, USERPROFILE), so `MOCK_MEDIA` and the keys are passed by name (D13).
- **`chat_schema(tool, hidden)`**: an MCP `Tool` → the Chat Completions `{"type": "function", ...}`
  format: drops the `hidden` arguments (`run_id`) from properties and required, drops an empty
  `required`, strips Pydantic titles with Phase 3's `_without_titles()`.
- **`tool_message(result)`**: an MCP `CallToolResult` → the string of a tool message.
  `isError` → raise `tools.ToolFailed(text)`; structured content → `tools.ok_json(...)` (adds
  `"ok": true`, the exact string the direct call returns); else the text.
- **`McpToolbox`**: `connect(server, mode="legacy")` is a normal (sync) context manager:
  `start_blocking_portal()` runs an event loop in a background thread,
  `portal.wrap_async_context_manager(Client(server, mode=mode))` enters the async client in it.
  Methods: `list_tools()`, `call()`, `read_text(uri)`, `schemas(names, hidden)`,
  `tool_caller(**bound)` → the `CallTool` function `run_loop()` takes (D10, D11).

### `loop.py`: `--mcp`

`main()` now ends in `produce(run_dir, script, status, tool_schemas, call_tool, args, via)`, the
part both paths share (prompt, `run_loop`, `agent_log.jsonl`, `agent_messages.json`, summary).
- direct: as in Phase 3 (`tools.schemas()`, `tools.call_tool(ctx, ...)`).
- `--mcp` (`produce_over_mcp()`): start the server as a child; `--from` → `create_run` over MCP
  with the source script (so the server validates it); `--run DIR` → `run_id = DIR.name`; read
  the script and the status **as resources**; `box.schemas(tools.TOOLS, hidden=["run_id"])`
  (only the 3 media tools, like Phase 3); `box.tool_caller(run_id=run_id)`.

---

## 2. stdio: what "nothing may print to stdout" means here, measured

stdout is the JSON-RPC channel: one JSON message per line. Anything else on it is garbage to
the host. Three experiments (a scratch server whose tool prints, raw pipes and the SDK client):

| what printed | where it went | effect |
|---|---|---|
| `print(..., flush=True)` inside a tool, while serving | **stderr** | none: the SDK points fd 1 at stderr while serving |
| a child process (like ffmpeg) printing and reading stdin | stdout → **stderr**; stdin read `''` (EOF) | none: children inherit the diverted fds |
| `print("hello from import time")` before `mcp.run()` | **stdout**, before the initialize answer | the SDK client logs `Failed to parse JSONRPC message from server` + `Invalid JSON: expected value at line 1 column 1 [type=json_invalid, input_value='hello from import time\r', ...]` and carries on |
| `print(...)` (no flush) inside a tool | **stdout, at exit** | E3: the real leak, fixed |

Notes: the `\r` in that error is Windows' `\r\n` line ending in text mode. Our own code never
prints in the server path (`journal()` and `make.py` print, but the server doesn't use them),
and the server logs only to stderr. A host saves the server's stderr to a log file (§4).

---

## 3. The real JSON-RPC messages

Captured with a 25-line proxy placed between the official SDK client and the server (it logs
every line in both directions and passes it on). Lines are verbatim; the only changes: the long
temp path is shortened to `<tmp>`, and two long answers are cut where marked.

### Protocol 2025-11-25 (`mode="legacy"`): the classic handshake

```
C->S {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"mcp","version":"0.1.0"}}}
S->C {"jsonrpc":"2.0","id":1,"result":{"capabilities":{"prompts":{"listChanged":false},"resources":{"listChanged":false,"subscribe":false},"tools":{"listChanged":false}},"instructions":"Makes ~30-second vertical short videos (YouTube Shorts, Reels, TikTok). Flow: create_run with a script you write (or write_script) -> generate_image and text_to_speech for every scene -> render_video, which saves final.mp4 in the run folder.","protocolVersion":"2025-11-25","serverInfo":{"name":"shortsmith","version":""}}}
C->S {"jsonrpc":"2.0","method":"notifications/initialized"}
C->S {"jsonrpc":"2.0","id":2,"method":"tools/list"}
S->C {"jsonrpc":"2.0","id":2,"result":{"tools":[{"description":"Make the vertical 9:16 image for one scene, from the script's visual prompt.\nSaves scenes/scene_NN.png. An image that already exists is reused, not made again.","inputSchema":{"properties":{"run_id":{"description":"The run's id, as returned by create_run or write_script.","pattern":"^[A-Za-z0-9_-]{1,64}$","title":"Run Id","type":"string"},"scene":{"description":"Scene number (idx) from the script, starting at 1.","minimum":1,"title":"Scene","type":"integer"}},"required":["run_id","scene"],"type":"object","title":"generate_imageArguments"},"name":"generate_image","outputSchema":{"properties":{"scene":{"title":"Scene","type":"integer"},"path":{"title":"Path","type":"string"},"reused":{"title":"Reused","type":"boolean"}},"required":["scene","path","reused"],"type":"object","title":"ImageResult"}}, ...4 more tools... ]}}   (6,628 characters in total)
```
What to notice: requests have an `id`, the answer quotes it; a **notification** (`notifications/initialized`)
has no id and gets no answer. The capabilities say which primitives the server has.
`listChanged: false` = the tool list never changes while running.

The rest of the session (resources, prompts, then calls):
```
C->S {"jsonrpc":"2.0","id":3,"method":"resources/list"}
S->C {"jsonrpc":"2.0","id":3,"result":{"resources":[{"description":"Every run, newest first: id, title, language, scenes and whether final.mp4 exists.","mimeType":"application/json","name":"runs_list","uri":"shortsmith://runs"}]}}
C->S {"jsonrpc":"2.0","id":4,"method":"resources/templates/list"}
S->C {"jsonrpc":"2.0","id":4,"result":{"resourceTemplates":[{"description":"A run's script.json: title, hook, scenes (narration, visual prompt, motion), cta, language.","mimeType":"application/json","name":"run_script","uriTemplate":"shortsmith://runs/{run_id}/script"},{"description":"What a run already has: scenes with an image, scenes with a voiceover, and whether final.mp4 exists.","mimeType":"application/json","name":"run_status","uriTemplate":"shortsmith://runs/{run_id}/status"}]}}
C->S {"jsonrpc":"2.0","id":5,"method":"prompts/list"}
S->C {"jsonrpc":"2.0","id":5,"result":{"prompts":[{"arguments":[{"name":"topic","required":true},{"name":"language","required":false}],"description":"Make a ~30-second vertical short video about a topic: script, images, voice, final.mp4.","name":"make_short","title":"Make a short video"}]}}
C->S {"jsonrpc":"2.0","id":6,"method":"prompts/get","params":{"name":"make_short","arguments":{"topic":"Registan at night","language":"uz"}}}
S->C {"jsonrpc":"2.0","id":6,"result":{"description":"Make a ~30-second vertical short video about a topic: script, images, voice, final.mp4.","messages":[{"content":{"text":"Make a ~30-second vertical short video about: Registan at night\nLanguage: Uzbek (Latin script) (language code 'uz').\n\n1. Write the script yourself and start the run with create_run. Write it as this brief says:\n---\nYou write scripts for vertical short videos ... (the whole writer prompt) ...","type":"text"},"role":"user"}]}}
```

**A tool error (the script broke a rule)**: an ordinary *result* with `isError: true`, so the model reads it:
```
C->S {"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"create_run","arguments":{"script":{"title":"Cozy Autumn Coffee in Tashkent","hook":"This hook is far too long to fit into two short seconds of video","scenes":[{"idx":1,"narration":"Craving the ultimate cozy autumn vibe? Step inside this hidden Tashkent gem.", ...}, ...], ...}}}}
S->C {"jsonrpc":"2.0","id":7,"result":{"content":[{"text":"Error executing tool create_run: 1 validation error for create_runArguments\nscript.hook\n  Value error, hook has 14 words, max 10 (it must fit ~2 seconds) [type=value_error, input_value='This hook is far too lon... short seconds of video', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.14/v/value_error","type":"text"}],"isError":true}}
```

**A successful call** (the same script with its real hook, then a media tool):
```
C->S {"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"create_run","arguments":{"script":{ ...the whole script... }}}}
S->C {"jsonrpc":"2.0","id":8,"result":{"content":[{"text":"{\n  \"run_id\": \"2026-10-22_184118\",\n  \"folder\": \"<tmp>\\\\root2\\\\runs\\\\2026-10-22_184118\",\n  \"title\": \"Cozy Autumn Coffee in Tashkent\",\n  \"language\": \"en\",\n  \"scenes\": 6\n}","type":"text"}],"isError":false,"structuredContent":{"run_id":"2026-10-22_184118","folder":"<tmp>\\root2\\runs\\2026-10-22_184118","title":"Cozy Autumn Coffee in Tashkent","language":"en","scenes":6}}}
C->S {"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"generate_image","arguments":{"run_id":"2026-10-22_184118","scene":1}}}
S->C {"jsonrpc":"2.0","id":10,"result":{"content":[{"text":"{\n  \"scene\": 1,\n  \"path\": \"scenes/scene_01.png\",\n  \"reused\": false\n}","type":"text"}],"isError":false,"structuredContent":{"scene":1,"path":"scenes/scene_01.png","reused":false}}}
```
The result carries the data twice: as text (for hosts that only show text) and as
`structuredContent` matching the tool's `outputSchema` (for programs). Our agent uses the second.

**More tool errors** (all `isError: true` results):
```
S->C {"jsonrpc":"2.0","id":9,"result":{"content":[{"text":"Error executing tool render_video: FileNotFoundError: not ready to render: no image for scenes [1, 2, 3, 4, 5, 6]; no voiceover for scenes [1, 2, 3, 4, 5, 6]","type":"text"}],"isError":true}}
S->C {"jsonrpc":"2.0","id":11,"result":{"content":[{"text":"Error executing tool generate_image: 1 validation error for generate_imageArguments\nscene\n  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.14/v/greater_than_equal","type":"text"}],"isError":true}}
S->C {"jsonrpc":"2.0","id":12,"result":{"content":[{"text":"Error executing tool generate_image: 1 validation error for generate_imageArguments\nrun_id\n  String should match pattern '^[A-Za-z0-9_-]{1,64}$' [type=string_pattern_mismatch, input_value='../../etc', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.14/v/string_pattern_mismatch","type":"text"}],"isError":true}}
S->C {"jsonrpc":"2.0","id":13,"result":{"content":[{"text":"Error executing tool generate_image: ValueError: no run 'nope': start one with create_run or write_script","type":"text"}],"isError":true}}
```
(id 12: `run_id: "../../etc"`, a path-traversal attempt, stopped by the schema's pattern.)

**A resource, and a resource error**: the error is a JSON-RPC `error`, not a result:
```
C->S {"jsonrpc":"2.0","id":14,"method":"resources/read","params":{"uri":"shortsmith://runs/2026-10-22_184118/status"}}
S->C {"jsonrpc":"2.0","id":14,"result":{"contents":[{"mimeType":"application/json","text":"{\"images\":[1],\"voiceovers\":[],\"final\":false}","uri":"shortsmith://runs/2026-10-22_184118/status"}]}}
C->S {"jsonrpc":"2.0","id":15,"method":"resources/read","params":{"uri":"shortsmith://runs/nope/status"}}
S->C {"jsonrpc":"2.0","id":15,"error":{"code":-32602,"message":"no run 'nope': start one with create_run or write_script","data":{"uri":"shortsmith://runs/nope/status"}}}
```
Why two kinds of errors: a tool error is information **for the model** (it can fix its call and
retry), so it travels as a normal result. A protocol error (unknown method, bad params, a
resource that doesn't exist) is for the **program** that sent the request.

### Protocol 2026-07-28 (`mode="auto"`, the SDK's default): no handshake

```
C->S {"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"mcp","version":"0.1.0"},"io.modelcontextprotocol/clientCapabilities":{}}}}
S->C {"jsonrpc":"2.0","id":1,"result":{"cacheScope":"private","capabilities":{"prompts":{"listChanged":true},"resources":{"listChanged":true,"subscribe":true},"tools":{"listChanged":true}},"instructions":"Makes ~30-second ...","resultType":"complete","supportedVersions":["2026-07-28"],"ttlMs":0,"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"shortsmith","version":""}}}}
C->S {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"render_video","arguments":{"run_id":"2026-10-22_184122"},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"mcp","version":"0.1.0"},"io.modelcontextprotocol/clientCapabilities":{}}}}
S->C {"jsonrpc":"2.0","id":4,"result":{"content":[{"text":"Error executing tool render_video: FileNotFoundError: not ready to render: ...","type":"text"}],"isError":true,"resultType":"complete","_meta":{"io.modelcontextprotocol/serverInfo":{"name":"shortsmith","version":""}}}}
```
Differences: `server/discover` replaces initialize/initialized; every request carries the
protocol version and client info in `_meta`, so the server keeps no per-connection state
(easier to scale over HTTP); results gain `resultType` and caching hints (`ttlMs`, `cacheScope`).
Tools, resources and prompts look the same. Our agent uses `mode="legacy"` so what it does is
the classic flow the lesson explains (D12); both work with this server.

---

## 4. Claude Desktop: config and steps for the owner

**Not done by the build session** (the brief: don't touch the owner's config). What *was*
verified: the exact command below, started by the SDK client from `C:\Windows\System32` with
the SDK's minimal environment, came up in 1.8 s; its startup line said
`root D:\All codes\shortsmith | images: placeholders (MOCK_MEDIA) | LLM provider: gemini (key set) | ffmpeg: C:\Users\doniy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg.EXE`
(so `--directory` handles the space, `.env` is found relative to it, and ffmpeg is on PATH), and
tools/list + resources/read worked.

### The snippet

```json
{
  "mcpServers": {
    "shortsmith": {
      "command": "C:\\Users\\doniy\\AppData\\Local\\Microsoft\\WinGet\\Packages\\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\\uv.exe",
      "args": [
        "--directory", "D:\\All codes\\shortsmith",
        "run", "--env-file", ".env",
        "python", "-m", "shortsmith.mcp_server"
      ]
    }
  }
}
```
- JSON needs every backslash doubled. The space in `D:\All codes\shortsmith` needs nothing:
  each array element reaches uv as one argument (no shell splits it).
- `command` is uv's absolute path (from `where uv`): a GUI app's PATH can differ from a terminal's.
- `--directory` makes uv run inside the repo (it finds `pyproject.toml` and `.venv` there), and
  relative paths after it, like `--env-file .env`, resolve against it. So the keys stay in
  `.env`; **never put keys into this config file**. The server also `chdir`s to the repo itself.
- No `"env"` block needed: `MOCK_MEDIA` defaults to mock (placeholders, $0) when unset, and
  `.env` sets `LLM_PROVIDER=gemini` for `write_script`.
- If the file already has content (other servers, a `"preferences"` block), add only the
  `"shortsmith": {...}` entry inside the existing `"mcpServers"` object, or add `"mcpServers"`
  next to the other top-level keys. One wrong comma and Claude ignores the whole file.

### Steps

1. **Finish or pause your Code sessions first.** These Code sessions run inside the same Claude
   app, and step 4 restarts it.
2. Open the config: Claude app → **Settings → Developer → Edit Config**. That opens the right
   `claude_desktop_config.json`. On this machine it is
   `C:\Users\doniy\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\claude_desktop_config.json`
   (the usual `%APPDATA%\Claude\` folder only holds a marker file here: this install keeps its
   data under `Claude-Data`). The file exists already.
3. Add the snippet (see the merge note above) and save.
4. **Quit Claude completely**: right-click its icon in the system tray (bottom right) → Quit.
   Closing the window isn't enough. Start Claude again.
5. Check: Settings → Developer should list **shortsmith** as running. In a new **Chat** (not
   Code), the tools menu ("+" / "Search and tools"; the names vary between app versions) should
   show shortsmith with 5 tools.
6. Make a video. Either pick the prompt (in the "+" menu: shortsmith → **Make a short video**,
   topic e.g. `cozy autumn coffee in Tashkent`, language `en`), or type:
   > Make a 30-second vertical short video about cozy autumn coffee in Tashkent, in English,
   > with the shortsmith tools. Write the script yourself and start it with create_run.

   Claude asks permission for each tool the first time (Allow once / Always allow for this chat).
7. What to expect: `create_run` (maybe rejected once, e.g. too few words, then fixed: our
   validator talking to Claude), then 6 × `generate_image` and 6 × `text_to_speech` (about 10 s
   in total), then `render_video` (about 20 s), then an answer with the folder; the video is
   `D:\All codes\shortsmith\runs\<run_id>\final.mp4`. Cost: $0 on our side (placeholder images,
   free voice, local ffmpeg); Claude's own tokens are your Claude plan. Take the screenshot.
   Afterwards you can attach `shortsmith://runs` from the "+" menu (a resource) to see every run.
8. If it fails, read the logs in `C:\Users\doniy\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\logs\`:
   `mcp-server-shortsmith.log` (should hold our server's stderr, starting with the
   `shortsmith MCP server | root ...` line) and `mcp.log` (Claude's side of the connection).
   (These two file names are the usual ones; they appear only after the first start.)
   - no shortsmith in Settings → Developer: the JSON is invalid (check commas and doubled backslashes);
   - `spawn ... ENOENT`: the uv path is wrong (`where uv` in a terminal gives the right one);
   - `ffmpeg: NOT FOUND` in the startup line: Claude was started before ffmpeg was on PATH; restart it;
   - to test outside Claude, in a terminal at the repo root:
     `uv run --env-file .env python -m shortsmith.mcp_server` should print one log line and then
     wait silently (it's waiting for JSON on stdin; Ctrl+C to stop).
   - optional: the official MCP Inspector (`npx @modelcontextprotocol/inspector`) is a web UI
     that connects to the same command (not used in this build).
9. A possible snag: `write_script` calls Gemini's free tier, which took 13–254 s per call in
   Phase 3. A host may give up on a tool call that takes that long (the TypeScript SDK that
   desktop hosts use has a default request timeout, 60 s in the versions I know). If
   `write_script` times out, ask Claude to write the script itself (`create_run`); that's the
   better demo anyway.

---

## 5. The live demos (Gemini 3.5 Flash-Lite, free tier, MOCK_MEDIA=1)

### Demo 1: our agent makes final.mp4 through the MCP server (the "done when" run)

```
uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855 --mcp
```
Console (verbatim; the first two lines are the server's own stderr log, i.e. another process):
```
2026-10-22 18:46:08,414 INFO shortsmith.mcp: shortsmith MCP server | root D:\All codes\shortsmith | images: placeholders (MOCK_MEDIA) | LLM provider: gemini (key set) | ffmpeg: C:\Users\doniy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.EXE
2026-10-22 18:46:08,432 INFO shortsmith.mcp: new run 2026-10-22_184608: 'Cozy Autumn Coffee in Tashkent' (en, 6 scenes)
runs\2026-10-22_184608 | 'Cozy Autumn Coffee in Tashkent' | tools: generate_image, text_to_speech, render_video | over MCP: server 'shortsmith', protocol 2025-11-25, started in 1.8 s
turn  1 |   539 in  180 out |  19.0 s | generate_image(1), generate_image(2), generate_image(3), generate_image(4), generate_image(5), generate_image(6), text_to_speech(1), text_to_speech(2), text_to_speech(3), text_to_speech(4), text_to_speech(5), text_to_speech(6)
turn  2 |  1313 in   10 out |  14.8 s | render_video()
turn  3 |  1377 in   17 out |  15.4 s | answer: "I have successfully rendered the final video for 'Cozy Autumn Coffee in Tashkent'." | STOP: done

stopped: done after 3 turns | 3229 in + 207 out tokens | ledger $0.001486 list price | 78 s
runs\2026-10-22_184608\final.mp4 | log: runs\2026-10-22_184608\agent_log.jsonl
```

**Compared with Phase 3's direct run of the same script** (`runs/2026-10-22_175609`, demo 1 there):

| | Phase 3, direct | Phase 4, over MCP |
|---|---|---|
| turns | 3 (12 parallel calls → render → answer) | 3, the same pattern |
| prompt tokens per turn | 539 / 1,313 / 1,377 | **539 / 1,313 / 1,377** (identical) |
| completion tokens | 180 / 10 / 21 | 180 / 10 / 17 (only the closing sentence differs) |
| list price | $0.001496 | $0.001486 (billed $0) |
| LLM wait per turn | 108.5 / 189.8 / 41.5 s | 19.0 / 14.8 / 15.4 s |
| tool time | images 0.28 s, voice 9.2 s, render 21.4 s | images 0.18 s, voice 9.5 s, render 19.6 s |
| wall time | 371 s | 78 s (incl. 1.8 s server start) |
| final.mp4 | 1080x1920, 902 frames, 30.07 s, 2,833,939 bytes | 1080x1920, **902 frames**, 30.07 s, 2,834,915 bytes |

- **Same video.** The 6 images are byte-identical (placeholders are deterministic), `subtitles.srt`
  is byte-identical, the frame count is the same. The ~1 KB size difference: the 6 voice MP3s
  differ in bytes (edge-tts synthesized them again) while their durations and word timings are
  identical (hence the identical subtitles and tool results).
- **Same conversation.** The user message and **all 13 tool messages** in `agent_messages.json` are
  byte-identical between the two runs. The model could not tell its tools had moved to another
  process.
- **Same prompt tokens, although the schemas differ.** Over MCP the parameters have no
  `"additionalProperties": false` (MCPServer's argument models don't forbid extra keys), so by
  tiktoken the three schemas are 287 instead of 305 tokens. Gemini still counted 539 on turn 1
  both times: Gemini's OpenAI-compatible endpoint evidently drops `additionalProperties` when it
  converts the schemas, so the model saw the same thing.
- **The time difference is Gemini's, not MCP's.** The whole difference (371 s vs 78 s) is LLM
  waiting: the free tier was much less busy at 18:46 than at 17:56 (Phase 3's E2: the same
  request varies up to 7x). MCP's own cost, measured separately (median of 50 calls of
  `generate_image` on an existing image): **0.16 ms direct vs 2.41 ms over stdio** (JSON encode,
  pipe, a worker thread, loading the run's context from disk, and back). For 13 calls per video
  that is ~30 ms. Server start + handshake: 1.8–2.1 s (Python start and imports; the handshake
  itself is ~3 ms), shutdown 0.24 s.

### Demo 2: an injected failure through MCP

```
uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112757 --mcp --fail-first text_to_speech
```
Console (verbatim; lines 1, 2 and 4 are the server process's log):
```
2026-10-22 18:48:30,732 INFO shortsmith.mcp: shortsmith MCP server | root D:\All codes\shortsmith | images: placeholders (MOCK_MEDIA) | LLM provider: gemini (key set) | ffmpeg: C:\...\ffmpeg.EXE | first call fails on purpose: text_to_speech
2026-10-22 18:48:30,750 INFO shortsmith.mcp: new run 2026-10-22_184830: 'Samarqand Registoni kechqurun' (uz, 6 scenes)
runs\2026-10-22_184830 | 'Samarqand Registoni kechqurun' | tools: generate_image, text_to_speech, render_video | over MCP: server 'shortsmith', protocol 2025-11-25, started in 1.6 s | first call fails on purpose: text_to_speech
2026-10-22 18:48:54,239 INFO mcp.server.mcpserver.server: Tool 'text_to_speech' failed: 'Error executing tool text_to_speech: ConnectionError: the service closed the connection without an answer'
turn  1 |   616 in  180 out |  23.2 s | generate_image(1), generate_image(2), generate_image(3), generate_image(4), generate_image(5), generate_image(6), text_to_speech(1) FAILED, text_to_speech(2), text_to_speech(3), text_to_speech(4), text_to_speech(5), text_to_speech(6)
turn  2 |  1375 in   16 out |  16.9 s | text_to_speech(1)
turn  3 |  1447 in   10 out |  15.8 s | render_video()
turn  4 |  1511 in   20 out |  27.8 s | answer: 'The video for Samarqand Registoni has been successfully created and saved as final.mp4.' | STOP: done

stopped: done after 4 turns | 4949 in + 226 out tokens | ledger $0.002050 list price | 111 s
runs\2026-10-22_184830\final.mp4 | log: runs\2026-10-22_184830\agent_log.jsonl
```

The path of the failure, end to end:
1. The server process (started with `--fail-first text_to_speech`) raised the injected
   `ConnectionError` inside `run_tool()`; `media_tool()` turned it into
   `ToolError("ConnectionError: the service closed the connection without an answer")`.
2. On the wire: a result with `"isError": true` and the text
   `Error executing tool text_to_speech: ConnectionError: the service closed the connection without an answer`.
3. Our client: `tool_message()` raised `ToolFailed(<that text>)`; `run_loop()` caught it like any
   tool failure and wrote message `[9]` of `agent_messages.json`:
   `{"role": "tool", "tool_call_id": "call_147449", "content": "{\"ok\": false, \"error\": \"Error executing tool text_to_speech: ConnectionError: the service closed the connection without an answer\"}"}`
4. The model's next turn: `text_to_speech` with `{"scene":1}`, **only the failed call**, then
   render, then a sentence. Exactly Phase 3 demo 2's behaviour (4 turns), now across a process boundary.

| | Phase 3 demo 2 (direct) | Phase 4 demo 2 (MCP) |
|---|---|---|
| prompt tokens per turn | 616 / 1,366 / 1,438 / 1,502 | 616 / **1,375 / 1,447 / 1,511** |
| total tokens, list price | 5,148, $0.002046 | 5,175, $0.002050 |
| LLM wait | 462 s | 84 s |
| final.mp4 | 863 frames, 28.77 s | **863 frames**, 28.77 s, `subtitles.srt` byte-identical |

The **+9 prompt tokens** from turn 2 on are the SDK's prefix `Error executing tool text_to_speech: `
(37 characters) inside the one error message, re-sent on every later turn: the only place where
the model's conversation differs from Phase 3. Small, but a nice example of "every character in
the history is paid again on every turn".

### All Phase 4 LLM calls (from `runs/ledger.jsonl`)

7 calls, all `step="agent"`, Gemini 3.5 Flash-Lite: demo 1 (3 calls, 13:46–13:47 UTC,
$0.001486) and demo 2 (4 calls, 13:48–13:50 UTC, $0.002050) = **$0.003536 list price, billed $0**.
Everything else (server tests, wire captures, the launch-command check, the overhead
measurement) used no LLM. Ledger total after the phase: $0.0224822 list price.

---

## 6. Errors hit, how they were found, how they were fixed

**E1. pytest crashed: `Windows fatal exception: access violation` in
`win32comext\taskscheduler\test\test_addtask.py`.** The first `uv run pytest` after adding mcp.
The traceback showed pytest *collecting pywin32's own test files* (pywin32 came with mcp).
Cause: my shell's working directory had drifted into `.venv\Lib\site-packages` (an earlier
`cd` there to read the SDK's source), so pytest found no `pyproject.toml`, used that folder as
its root, ignored `testpaths = ["tests"]` and collected everything below it. Fix: run from the
repo root (`Set-Location "D:\All codes\shortsmith"` first). Nothing wrong in the code. Lesson:
`testpaths` and the rest of the pytest config only apply when pytest starts where it can find them.

**E2. A stray `print()` in a tool leaked onto stdout at shutdown (found by our own test).**
`test_stdout_carries_only_json_rpc_even_when_a_tool_prints` failed on its first run:
`json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` for the line
`'a stray print() inside a tool'`. But the scratch experiment in §2 had shown such a print going
to stderr. The difference: the experiment used `flush=True`, the test didn't. Python keeps
`print()` output in a memory buffer when stdout is a pipe (block buffering); the SDK's diversion
works at the file-descriptor level (fd 1 → stderr) only *while serving*. The buffered line was
written when the interpreter exited, after `stdio_server()` had already pointed fd 1 back at the
real pipe. Order of the output proved it: the initialize answer, then the stray line, at the end.
Fix: `sys.stdout.reconfigure(line_buffering=True)` in `main()`: every printed line is flushed at
once, i.e. while fd 1 still points at stderr. Mutation check: with that line removed, the test
fails with `AssertionError: assert ['a stray pri...ide a tool\n'] == []`; with it, it passes.
(It can't be fixed by setting `sys.stdout = sys.stderr`: the SDK finds the channel through
`sys.stdout`, so it would then serve JSON-RPC on stderr.)

**E3 (same test run). The tools/call got no answer at all.** The first version of the test sent
all four requests and closed stdin at once (`subprocess.run(input=...)`). The answers to
initialize came back, the tool even ran (the print happened), but the tools/call answer never
came. Cause: in stdio, **closing stdin is how the host tells the server to shut down**; the server
stopped and dropped the request still running in its worker thread. Fix (in the test): keep stdin
open, read each answer before sending the next request, close stdin only at the end. The same
rule matters for any host: shut down only after the last answer.

**E4. The shell edit of tools.py failed** (`unexpected EOF while looking for matching `''`): a
Python heredoc inside bash with `'''` strings. Tooling only; the edits were made with the editor instead.

**E5 (cosmetic). Unreadable logs.** MCPServer installs a `rich` log handler on stderr; when stderr
isn't a terminal (a host's log file) it wraps every message into a narrow column, one long line
becoming 13 short ones. `main()` replaces it with plain one-line logging (`force=True`).

**Not errors, but differences worth knowing:**
- *Argument errors read differently from Phase 3.* MCPServer validates the arguments against the
  input schema **before** our function runs, and reports a failure as pydantic's own text
  (`1 validation error for generate_imageArguments\nscene\n  Input should be greater than or
  equal to 1 [type=..., input_value=0, ...]\n    For further information visit https://...`),
  ~250 characters with a URL, where Phase 3 said `invalid arguments: scene: Input should be
  greater than or equal to 1`. Errors from *our* code (not ready to render, no such scene, the
  injected ConnectionError) keep Phase 3's words via `describe_error()`, with the SDK's
  `Error executing tool <name>: ` in front.
- *Any other exception's text would be hidden.* In mcp 2.x a tool that raises an ordinary
  exception produces only `Error executing tool <name>` (the text "stays on the server": it may
  contain secrets). Only `ToolError` keeps its message. That is why `media_tool()` converts every
  exception into `ToolError(describe_error(e))`; without it the model would read nothing useful.

---

## 7. Design decisions and why

**D1. The registry is published as it is, with generated signatures.** `MCPServer` builds a tool's
input schema from its Python function's signature. So `media_tool(t)` makes, for every tool in
`tools.TOOLS`, a function whose `__signature__` is `run_id` + the fields of the tool's args model
(`Annotated[field.annotation, field]` keeps `ge=1` and the description) and whose return
annotation is the registry function's result model. Name and description come from the
registry too. One source for both front-ends: a new tool added to the registry appears in the
Phase 3 loop *and* in Claude Desktop, with no second definition to drift. Alternatives
considered: three hand-written `@mcp.tool()` wrappers (clearer to read, but the argument types
and descriptions would be written twice), and the SDK's low-level `Server` with my own
`tools/list` and `tools/call` handlers (full control over validation messages, but it skips what
Phase 4 is meant to teach: the decorator-based server everybody uses). The two new tools use the
plain `@mcp.tool()` decorator, so the lesson shows both ways.

**D2. Every media call names its run, and the server keeps no run state in memory.** A Phase 3
`ToolContext` was bound to one run by the program. An MCP server serves whatever conversation
talks to it, so the run comes with each call, and `ToolContext.for_run()` is loaded from disk
every time (~2 ms). Consequences: a server restart loses nothing; Claude Desktop and our agent can
each run their own server process on the same `runs/` folder; and the per-scene voice had to move
to disk: `text_to_speech` now writes `audio/scene_NN.json` (the word timings) next to the MP3, and
`for_run()` picks these up when there is no complete `voiceovers.json`. (Before, a half-voiced
run lived only in memory.) Tested: two voices, a fresh client, two more voices, render works.

**D3. `run_id` is a folder name, checked twice.** The schema's pattern `^[A-Za-z0-9_-]{1,64}$`
rejects `../../etc` before anything runs (captured in §3); `run_dir()` checks it again because
resources take the id from a URI, not through a schema. No tool takes a path (as in Phase 3).

**D4. Tool errors are results; resource errors are protocol errors.** For tools, the model must
be able to read the failure and react, so every exception becomes `ToolError(describe_error(e))`
→ `isError: true` with readable text. For resources, the *program* asked, so a missing run is a
JSON-RPC error (`-32602`). `describe_error()` (moved from loop.py to tools.py) writes the same
words for both front-ends.

**D5. Structured results, and "ok" added back on the client.** The media tools return their
Pydantic result models, so MCPServer publishes an `outputSchema` and sends `structuredContent`
(plus the same JSON as text for hosts that only read text). MCP says success/failure with
`isError`, so the server doesn't send Phase 3's `"ok": true`. A Chat Completions tool message is
only a string, so the client adapter puts it back (`tools.ok_json()`, the function `call_tool()`
uses too): that's why the model's 13 tool messages are byte-identical in §5.

**D6. `create_run` takes the whole Script; our Pydantic model is the gatekeeper.** The host's model
(Claude) writes the script itself; MCPServer validates it against `Script` (shape from the
schema, rules from our validators) before `create_run` runs, and a broken script comes back as
an error result with the reason, e.g. `hook has 14 words, max 10`. The input schema has no
field descriptions (the Script model has comments, not `Field(description=...)`, and adding
them would change the writer's `response_format` from Phase 1), so the tool's **description**
carries the rules, built from the validator's constants so they can't disagree. `write_script` is
the alternative for hosts that shouldn't write (one Gemini call, in the ledger).

**D7. Resources: a list and two templates, and our agent reads them.** Read-only views of
`runs/`. Our agent's *program* (not the model) reads `shortsmith://runs/{id}/script` and
`.../status` to write the task message, which is exactly what resources are for: context the
application chooses to load. The model only gets the tools.

**D8. The prompt reuses the writer's brief.** `make_short` sends `llm.WRITER_SYSTEM_PROMPT` as part
of its message, so Claude writes scripts by the same rules as our Gemini writer, from one source.

**D9. stdio, and the server makes itself independent of how it is started.** `os.chdir(--root)`
(hosts start servers from their own folder), logs only to stderr, line-buffered stdout (E2), a
startup line that says what the server sees (root, mock or paid images, LLM provider and whether
its key is set (never the key), where ffmpeg is). stdio because Claude Desktop uses it for local
servers; the same `MCPServer` can serve Streamable HTTP with `mcp.run("streamable-http")` (Phase 8/9
if ever needed).

**D10. The loop stays synchronous; an anyio blocking portal bridges to the async client.**
`McpToolbox.connect()` runs the MCP client in an event loop on a background thread
(`start_blocking_portal()` + `portal.wrap_async_context_manager(Client(...))`), and every call from
the loop is `portal.call(client.call_tool, ...)`: it blocks like a normal function. So `run_loop()`
didn't change at all, and `main()` reads top to bottom. (Rejected: making the loop async, which
would change the Phase 3 exercise and every test.)

**D11. The program binds `run_id`; the model never sees it.** `chat_schema(..., hidden=["run_id"])`
removes it from what the model sees, and `tool_caller(run_id=...)` adds it to every call, last,
so a model that invents a `run_id` can't point a call at another run (tested). This keeps
Phase 3's guarantee (the program picks the run) and makes the comparison fair: the model saw the
same tools as in Phase 3.

**D12. Our agent uses `mode="legacy"`** (the 2025-11-25 initialize handshake). It's the flow
every MCP explanation shows and the one in §3; the SDK's default would silently switch to
2026-07-28. Both work with this server (tested).

**D13. The child gets only the environment it needs.** `stdio_client` passes a short safe list
(PATH, APPDATA, ...) on purpose: a server is someone else's code, it shouldn't see all your keys.
Our agent passes `MOCK_MEDIA`, `LLM_PROVIDER` and the three keys by name (`PASS_ENV`). Claude
Desktop gets them through `uv run --env-file .env` instead, so no key is ever written into its
config file.

**D14. The paid-media rule still holds over MCP.** The server never makes paid images unless the
server process itself sees `MOCK_MEDIA=0` (default mock), and the loop still refuses
`MOCK_MEDIA=0` without `--allow-paid-media`. Note: a host like Claude Desktop has no such flag,
so `MOCK_MEDIA=0` must never go into `.env` while the server is configured there.

**D15. `--fail-first` lives in the server process.** The failure must happen where the tool runs.
The loop's `--fail-first` is passed through as a server argument; the server remembers "already
failed" in a process-wide set (`FAILED`), because its `ToolContext`s are per call.

---

## 8. The plan's questions, answered from this build

**Q1. What's the difference between a tool and a resource (and a prompt)?** Who decides to use it.
- A **tool** is *model-controlled*: the server lists it with an input schema, the **model**
  decides to call it, and calling it does something (makes `scenes/scene_01.png`). Its result
  goes back into the model's context. Errors come back as results (`isError`) so the model can
  react. Evidence: in demo 1 the model chose 13 tool calls on its own; in demo 2 it read an
  `isError` result and retried.
- A **resource** is *application-controlled*: read-only data with a URI
  (`shortsmith://runs/{run_id}/status`). The **app** (or the user) decides to read it and what to
  do with it; reading changes nothing. Evidence: our agent's program read the script and status
  resources to build the task message; the model never saw a resource, only the text the program
  wrote from it. In Claude Desktop the user attaches a resource from the "+" menu. Errors are
  protocol errors (§3, id 15), for the program.
- A **prompt** is *user-controlled*: a template with arguments that the **user** picks (like a
  slash command, "Make a short video") and the host turns into messages for the model. Evidence:
  `prompts/get make_short {topic, language}` returned one user message with the writer's brief.
- Rule of thumb: if it has side effects or the model should decide when, it's a tool; if it's
  context to load, it's a resource; if it's a reusable way of asking, it's a prompt. (A host could
  of course *wrap* a resource in a tool so the model can read it; MCP just says whose decision it is.)

**Q2. Why is MCP better than every app writing its own plugins (M×N → M+N)?** Without a standard,
each of M AI apps (Claude Desktop, Cursor, ChatGPT, our own agent, ...) needs its own plugin for
each of N tools (video, GitHub, Postgres, ...): M×N integrations, each with its own schema
format, transport and error conventions. With one protocol, each app implements an MCP client
once and each tool an MCP server once: M+N. Evidence from this build: **one** server file, and two
completely different hosts used it unchanged: our own Python agent (Gemini) and, with a 10-line
JSON config, Claude Desktop (Claude). Neither host has a line of Shortsmith-specific code: our
agent discovered the tools with `tools/list` and converted them with a generic 15-line function
(`chat_schema`), and Claude Desktop does the same internally. The other direction counts too:
our agent's MCP client can use *any* MCP server (filesystem, GitHub, ...) with the same
`McpToolbox`. The price: a process boundary (2.4 ms per call here vs 0.16 ms in-process, 2 s to
start), JSON-RPC encoding, and one more thing to configure and debug (E2, E3).

**Other plan concepts, briefly, with evidence:**
- *Host / client / server*: the host is the app with the model (Claude Desktop; our `loop.py`);
  it runs one MCP client per server connection (`McpToolbox` holds one `mcp.Client`); the server
  offers the capabilities (`mcp_server.py`). Here host and client live in one process, the server in another.
- *Transports*: stdio (the host starts the server as a child; one JSON message per line on
  stdin/stdout; closing stdin = shutdown, E3; stdout must stay clean, E2) for local servers;
  Streamable HTTP (POST + optional server-sent events) for remote servers shared by many users.
  The messages are the same JSON-RPC either way.
- *Tool discovery*: nothing is hard-coded on the host side. `tools/list` returns names,
  descriptions and JSON schemas at runtime; `listChanged` in the capabilities says whether the
  list can change while connected (ours: false).

---

## 9. Tests

- `uv run pytest`: **154 passed** (~10 s; 117 before this phase + 37). Per file: test_ffmpeg 31,
  test_mcp_server 25, test_tools 21, test_loop 20, test_mcp_tools 11, test_tts 10, test_models 9,
  test_llm 8, test_ledger 8, test_make 7, test_images 4.
- `test_tools.py` (+1): three scenes voiced, one MP3 deleted: a fresh `for_run()` finds scenes 1
  and 2 from their `audio/scene_NN.json` (paths back to absolute, same word timings) and skips 3.
- How the server is tested: `McpToolbox.connect(mcp_server.mcp)` gives the official SDK `Client`
  connected to the server object in-process (initialize handshake and all, no subprocess), in a
  temporary `runs/` (`monkeypatch.chdir`). Voice and ffmpeg are faked as in Phase 3.
- `test_mcp_server.py` (25): the 5 tools, media descriptions = registry docstrings, schema = registry
  args + `run_id` with its pattern, output schema = result model, `create_run`'s schema and its rules
  in the description; `create_run` saves and returns an absolute folder; two runs in one second
  get different folders; 3 rule breaks → error results with the validator's words and no run
  started; `write_script` with a faked writer, and a writer that breaks the rules; a media call's
  tool message is byte-identical to a direct `call_tool`; voice survives a "restart" and the
  video renders; the "not ready" error with Phase 3's words; 4 bad calls (unknown run, `../../etc`,
  scene 0, scene 9); `--fail-first` once per process; the 3 resources and a missing run as an
  `MCPError`; the prompt's arguments and text; **a real subprocess over stdio** (list + call);
  **raw pipes, a tool that prints on purpose: every stdout line is JSON-RPC, nothing at exit,
  the print is in stderr** (E2's test, mutation-checked).
- `test_mcp_tools.py` (11): the model sees Phase 3's schemas (minus `additionalProperties`) with
  `run_id` hidden; empty schema for `render_video`; a missing tool → `KeyError`; structured / error
  / text results; the program's `run_id` wins over one the model invents; bad JSON raises like a
  direct call; **the Phase 3 loop with a scripted model reaches final.mp4 through MCP** (renders too
  early, reads the MCP error, fixes it); **`loop --from ... --mcp` end to end** (server in-process,
  scripted model): `create_run`, resources, 3 turns; `server_params()` passes only `PASS_ENV`.
- `uv run pytest exercises/phase-4`: 9 fail with `NotImplementedError` (by design); the same 9
  pass when `mcp_bridge.py` re-exports the real functions (checked in a scratch folder).
  `uv run pytest exercises`: 36 fail (16 + 11 + 9), no name clashes.

---

## 10. Open items

- **The Claude Desktop test** (4.3) is the owner's: config + restart + one chat + screenshot (§4).
- **SDK validation messages are long** (pydantic's text with a URL, §6). Fine for Claude; if it
  ever matters, a middleware or the low-level server could shorten them.
- **No progress notifications.** `render_video` takes ~20 s with no sign of life; MCP has
  `notifications/progress` (`ctx.report_progress` in a tool). Worth adding if a host times out.
- **`write_script` vs host timeouts** on a slow free tier (§4, step 9).
- **Parallel calls are still sequential** in our loop (Phase 3's D8). Over MCP they could be sent
  concurrently (MCPServer runs sync tools in worker threads); not needed at ~10 s of voice.
- **The MCP Inspector** was not used (the SDK client and raw pipes covered list/call/errors).
- CLAUDE.md's "Current progress" line is the teaching session's to update.
