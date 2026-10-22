# Phase 4 — MCP: one tool server, usable by any AI app

> Read at your own pace, ask anything. Parts 0–8 are the concepts. Parts 9+ (our server, the real
> JSON-RPC messages, Claude Desktop setup, the agent over MCP, numbers, errors) are completed from the
> build session's notes once the build is done.

---

## 0. The problem MCP solves

In Phase 3 our tools lived **inside our own program**. Only our agent could use them. Now imagine you
want the same tools in Claude Desktop, in an IDE assistant, in a teammate's agent, in ChatGPT…
Without a standard, every app needs its own plugin for every tool:

```
Without a standard (M × N)                 With MCP (M + N)
──────────────────────────                 ────────────────
App A ──┬── plugin for Tool 1              App A ─┐                 ┌─ Tool server 1
        ├── plugin for Tool 2              App B ─┼── speaks MCP ───┼─ Tool server 2
        └── plugin for Tool 3              App C ─┘                 └─ Tool server 3
App B ──┬── plugin for Tool 1
        ├── ...                            each app learns MCP once,
3 apps × 3 tools = 9 integrations          each tool is written once: 3 + 3 = 6
```

**MCP (Model Context Protocol)** is an open standard, introduced by Anthropic in late 2024 and now
supported by many AI apps and model providers. It defines **one way** for AI apps to discover and call
tools and data offered by other programs. The usual analogy: **USB-C for AI tools**. Any device works
with any charger because both follow the same plug standard.

That's the "**M × N → M + N**" idea from your plan, and the core answer to "why is MCP better than every
app writing its own plugins?"

---

## 1. The three roles: host, client, server

| Role | What it is | In our project |
|---|---|---|
| **Host** | the AI application the user works in; it owns the LLM and the conversation | **Claude Desktop**, or **our own agent** (`agents/loop.py`) |
| **Client** | a connector *inside* the host; **one client per server**, keeping a 1:1 connection | created by Claude Desktop from its config; in our agent, the MCP SDK's `ClientSession` |
| **Server** | a separate program that offers tools, data and prompt templates | **`shortsmith-mcp`** (`mcp_server.py`) |

The host never talks to the tools directly: it goes **host → its client → the server**. One host can
connect to many servers at once (files, GitHub, our video tools…), each through its own client.

---

## 2. What a server can offer: tools, resources, prompts

The single most important distinction in MCP is **who decides to use it**:

| Primitive | Controlled by | What it is | Shortsmith example |
|---|---|---|---|
| **Tool** | **the model** | an *action* the LLM decides to call (may change things, may cost money) | `generate_image`, `text_to_speech`, `render_video`, `create_run` |
| **Resource** | **the application / user** | *read-only data* the host can attach as context, addressed by a **URI** | a run's `script.json`, a list of runs |
| **Prompt** | **the user** | a *reusable message template* the user picks (often shown like a slash command) | "make a short video about {topic} in {language}" |

- A **tool** is like a button the model can press. It needs a **JSON Schema** for its arguments (as in
  Phase 3).
- A **resource** is like a file the app can open and show to the model. It has a URI like
  `runs://2026-10-22_112757/script`, and reading it should never change anything.
- A **prompt** is like a saved recipe the *user* starts on purpose. The server fills in the arguments
  and returns ready-made messages.

(There are also *client* features a server can use, like **sampling**: the server asks the host's LLM to
generate text. We don't need them now.)

---

## 3. The messages: JSON-RPC 2.0

Underneath, MCP is plain **JSON-RPC 2.0**: small JSON messages of three kinds:
- a **request** (has an `id`, expects an answer);
- a **response** (same `id`, with `result` or `error`);
- a **notification** (no `id`, no answer expected).

A connection always goes like this:
```
client → server   initialize        {protocolVersion, capabilities, clientInfo}
server → client   result            {protocolVersion, capabilities: {tools, resources, prompts}, serverInfo}
client → server   notifications/initialized
client → server   tools/list        → the tools with names, descriptions, input schemas
client → server   tools/call        {name: "generate_image", arguments: {...}}
server → client   result            {content: [{type: "text", text: "..."}], isError: false}
```

- **Capability negotiation:** in `initialize`, both sides say what they support, and only those
  features are used.
- **Discovery at runtime:** the host asks `tools/list`, so a server can add or change tools **without
  any change to the host**. That's what makes "write the tool once" work.
- **Tool errors** come back as a normal result with **`isError: true`**, so the model can read them and
  adapt, exactly like Phase 3's "errors as feedback". Protocol errors (malformed request, unknown method)
  are JSON-RPC `error` responses instead.
- **Protocol versions:** the flow above is protocol **2025-11-25**, the classic one most hosts speak. The
  newer **2026-07-28** drops the handshake: one `server/discover` call, then every request carries the
  version and client info itself (in `_meta`), so the server keeps no per-connection state, which makes it
  easier to scale over HTTP. Tools, resources and prompts look the same in both. Our agent uses the classic
  flow on purpose (section 12).

The build notes will show the real messages our server exchanged.

---

## 4. Transports: how the messages travel

| Transport | How | When |
|---|---|---|
| **stdio** | the host starts the server as a **child process**; messages go over its **stdin/stdout**, one JSON message per line | local servers on your own machine. **Claude Desktop uses this**, and so does ours |
| **Streamable HTTP** | the server is a web service; messages go over HTTP requests (optionally streamed) | remote/shared servers, many users, needs authentication (OAuth) |

**The stdio golden rule:** stdout *is* the message channel. A single stray `print("hello")` in the
server puts non-JSON text into the channel and breaks the connection. All logging must go to
**stderr**. Our code has `print()` calls in places (progress lines), so the build must route them away
from stdout.

---

## 5. MCP vs function calling (Phase 3): how they fit together

They're not competitors. They solve two different hops:

```
User ─► Host app ─(function calling)─► LLM          the model asks for a tool call   (Phase 3)
            │
            └─(MCP: tools/call)─► Tool server       the host asks the server to run it (Phase 4)
```

- **Function calling** is how *a model* tells *its app* "please call tool X". It's provider-specific
  (OpenAI format, Gemini format…).
- **MCP** is how *the app* discovers and calls tools that live in *another program*. It's the same for
  every app.

So our agent after Phase 4 does: `tools/list` (MCP) → convert the schemas to the Chat Completions format
→ send them to Gemini (function calling) → Gemini asks for a call → `tools/call` (MCP) → the result goes
back to Gemini. **The loop from Phase 3 stays the same**: only `call_tool` and the schemas now come
from an MCP client. That's why `run_loop()` takes `call_tool` as a parameter.

---

## 6. MCPServer (formerly FastMCP): writing a server in Python

The official Python SDK has a high-level server class that turns ordinary functions into MCP primitives
with decorators, much like our Phase 3 `@tool` registry. Most tutorials call it **FastMCP**
(`from mcp.server.fastmcp import FastMCP`). **In SDK version 2 (which we installed, mcp 2.3.0) it's
renamed `MCPServer`.** The old import now raises an error that says exactly that. Same decorators:

```python
from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="shortsmith")

@mcp.tool()
def generate_image(run_id: str, scene: int) -> str:
    """Make the vertical 9:16 image for one scene..."""        # becomes the tool description
    ...

@mcp.resource("runs://{run_id}/script")
def script(run_id: str) -> str: ...                             # read-only data, addressed by URI

@mcp.prompt()
def make_video(topic: str, language: str = "en") -> str: ...   # a template the user picks

mcp.run("stdio")                                                # stdio transport
```
(Lesson from the build: **check the installed version before trusting a tutorial.** Libraries in this
area move fast.)

- The **type hints and docstring** become the schema and the description (the same idea as our
  Pydantic-generated schemas in Phase 3).
- Our Phase 3 registry was deliberately shaped for this: name, docstring description, arguments
  model, function. So the build **reuses** those functions instead of copying them.

---

## 7. State: why the server needs run ids

In Phase 3 our tools were bound to **one** run folder (`ToolContext`). An MCP server can be used by
many conversations, and it can be **restarted** at any time (Claude Desktop restarts it when you
restart the app). So:
- tools take a **run id** (plus a scene number), not an invisible "current run";
- anything `render_video` needs (which voices exist, their word timings) must be **on disk**, not only in
  memory, so it survives a restart.

That's a general lesson: **a server should keep as little hidden state as possible.**

---

## 8. Security: an MCP server runs with your permissions

- A local stdio server is **a program running as you**: it can read and write anything you can. **Only
  install servers you trust**, like any other software.
- **Tool descriptions and results go straight into the model's context.** A malicious server can hide
  instructions there ("tool poisoning", a kind of prompt injection). A careless server can leak data.
- **Least privilege again:** our server only works inside `runs/`, takes run ids and scene numbers
  (never raw paths), and keeps paid media off unless explicitly enabled.
- **Hosts ask the user:** Claude Desktop shows a permission prompt before a tool runs. Don't click
  "always allow" for tools that can spend money or delete things.
- Remote (HTTP) servers add **authentication** (OAuth), because anyone on the internet could otherwise
  call them.

---

## 9. Our server (`src/shortsmith/mcp_server.py`)

**What it offers:**
| Kind | Name | What it does |
|---|---|---|
| tool | `generate_image(run_id, scene)` | Phase 3's tool, now with a run id |
| tool | `text_to_speech(run_id, scene)` | Phase 3's tool, now with a run id |
| tool | `render_video(run_id)` | Phase 3's tool, now with a run id |
| tool | **`create_run(script)`** | starts a run from a **whole script the host's model wrote itself** (e.g. Claude); our Pydantic `Script` model checks it first |
| tool | `write_script(topic, language)` | alternative: our own Gemini writer writes the script (one call, in the ledger) |
| resource | `shortsmith://runs` | every run, newest first: id, title, language, scenes, whether `final.mp4` exists |
| resource template | `shortsmith://runs/{run_id}/script` | a run's `script.json` |
| resource template | `shortsmith://runs/{run_id}/status` | what a run already has: `{"images": [1], "voiceovers": [], "final": false}` |
| prompt | `make_short(topic, language)` ("Make a short video") | a ready-made request including **our writer's rules** (`WRITER_SYSTEM_PROMPT`, reused, not copied) and the 3 steps |

**`MCPServer(name="shortsmith", instructions=...)`**: the `instructions` text goes to the host in the
initialize answer. Hosts may show it to their model: *"Flow: create_run with a script you write (or
write_script) → generate_image and text_to_speech for every scene → render_video…"*

**The Phase 3 tools are reused, not copied (D1): `media_tool(t)`**
- MCPServer builds a tool's input schema **from the Python function's signature**. So for every tool in
  Phase 3's `TOOLS` registry, `media_tool()` makes a small function whose **signature is generated**:
  `run_id` + the fields of the tool's arguments model (`scene: int ≥ 1` with its description). Its return
  type is the tool's result model (`ImageResult`…), which becomes an **`outputSchema`**.
- Its body:
  1. load the run's `ToolContext` **from disk** (`for_run(run_dir(run_id))`);
  2. call Phase 3's `run_tool()`;
  3. turn **any** exception into `ToolError(describe_error(e))`.
- **Why `ToolError`:** in SDK v2, a normal exception's text is hidden from the client (it might contain
  secrets), and the model would read only "Error executing tool …". `ToolError` keeps the message, so the
  model gets Phase 3's helpful words (*"not ready to render: no image for scenes [2, 3]…"*).
- **One source for both front-ends:** a new tool added to the registry appears in our loop *and* in
  Claude Desktop, with nothing written twice.

**Why every tool takes `run_id`, and nothing is kept in memory (D2):** a server serves whatever
conversation talks to it, and a host can restart it at any time. So each call names its run, and the
context is loaded from disk each time (~2 ms). The voice timings now also go to disk **per scene**
(`audio/scene_NN.json` next to each mp3), so a half-voiced run survives a restart. Consequences: a restart
loses nothing, and Claude Desktop and our agent can each run their own server process on the same `runs/`.

**`run_id` is a folder name, checked twice (D3).** The schema pattern `^[A-Za-z0-9_-]{1,64}$` rejects
`"../../etc"`, a **path-traversal** attempt, before anything runs. `run_dir()` checks it again, because
resources get the id from a URI, not through the schema.

**`create_run(script: Script)`: our validator talks to Claude (D6).**
- The host's model writes the script; MCPServer validates it against our `Script` model (hook length,
  4–8 scenes, order, **word counts**) *before* the function runs.
- A bad script comes back as an error result with the reason: *"hook has 14 words, max 10 (it must fit
  ~2 seconds)"*. The model fixes it and calls again.
- The tool's **description** (`CREATE_RUN`) is built from the **same constants** the validator uses, so
  the rules Claude reads and the rules we enforce can't disagree. That's Phase 1's lesson again.

**Tool errors vs resource errors (D4):**
- a **tool** error is a *result* with `isError: true`, for the **model** to read and react to;
- a missing **resource** is a JSON-RPC **protocol error** (code −32602), for the **program** that asked.

**`main()`: the server makes itself independent of how it's started (D9):**
- `os.chdir(--root)`: hosts start servers from their *own* folder, and our paths (`runs/`, the ledger) are relative.
- **Logs only to stderr**, one plain line each (the SDK's fancy console logger is replaced, E5).
- `sys.stdout.reconfigure(line_buffering=True)`: the fix for a real stdout leak (E2).
- One **startup line** that says what the server sees: root folder, mock or paid images, LLM provider and
  whether its key is set (**never the key itself**), where ffmpeg is. When something fails inside Claude
  Desktop, this line in its log file is the first thing to read.

---

## 10. The real JSON-RPC messages (captured on the wire)

The build session put a tiny **proxy** between the official SDK client and our server, logging every line
in both directions. Real messages, shortened:

**The handshake and discovery**
```
C→S {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"mcp",...}}}
S→C {"jsonrpc":"2.0","id":1,"result":{"capabilities":{"prompts":{...},"resources":{...},"tools":{"listChanged":false}},
                                       "instructions":"Makes ~30-second vertical short videos ...","serverInfo":{"name":"shortsmith"}}}
C→S {"jsonrpc":"2.0","method":"notifications/initialized"}                    ← no id: a notification, no answer
C→S {"jsonrpc":"2.0","id":2,"method":"tools/list"}
S→C {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"generate_image","description":"Make the vertical 9:16 image ...",
      "inputSchema":{"properties":{"run_id":{"pattern":"^[A-Za-z0-9_-]{1,64}$",...},"scene":{"minimum":1,"type":"integer",...}},
                     "required":["run_id","scene"],"type":"object"},
      "outputSchema":{"properties":{"scene":...,"path":...,"reused":...},...}}, ...4 more tools...]}}
```
- Requests carry an **`id`**, and the answer quotes it. `notifications/initialized` has **no id** and gets no answer.
- The **capabilities** say which primitives the server has. `listChanged: false` means the tool list never
  changes while connected.
- The whole `tools/list` answer is 2,061 tokens: what a host like Claude Desktop puts in front of its model.

**A successful call**
```
C→S {"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"generate_image","arguments":{"run_id":"2026-10-22_184118","scene":1}}}
S→C {"jsonrpc":"2.0","id":10,"result":{"content":[{"type":"text","text":"{\"scene\": 1, \"path\": \"scenes/scene_01.png\", \"reused\": false}"}],
      "isError":false,"structuredContent":{"scene":1,"path":"scenes/scene_01.png","reused":false}}}
```
The result carries the data **twice**: as **text** (for hosts that only show text) and as
**`structuredContent`** matching the `outputSchema` (for programs; our agent uses this one).

**Tool errors: normal results with `isError: true`**
```
S→C {..."result":{"content":[{"text":"Error executing tool render_video: FileNotFoundError: not ready to render: no image for scenes [1, 2, 3, 4, 5, 6]; ..."}],"isError":true}}
S→C {..."result":{"content":[{"text":"Error executing tool create_run: 1 validation error ... hook has 14 words, max 10 ..."}],"isError":true}}
S→C {..."result":{"content":[{"text":"Error executing tool generate_image: ... run_id String should match pattern ... input_value='../../etc'"}],"isError":true}}
```

**A resource, and a resource error: a protocol `error`, not a result**
```
C→S {"jsonrpc":"2.0","id":14,"method":"resources/read","params":{"uri":"shortsmith://runs/2026-10-22_184118/status"}}
S→C {"jsonrpc":"2.0","id":14,"result":{"contents":[{"mimeType":"application/json","text":"{\"images\":[1],\"voiceovers\":[],\"final\":false}",...}]}}
C→S {"jsonrpc":"2.0","id":15,"method":"resources/read","params":{"uri":"shortsmith://runs/nope/status"}}
S→C {"jsonrpc":"2.0","id":15,"error":{"code":-32602,"message":"no run 'nope': start one with create_run or write_script",...}}
```

**A prompt**
```
C→S {"jsonrpc":"2.0","id":6,"method":"prompts/get","params":{"name":"make_short","arguments":{"topic":"Registan at night","language":"uz"}}}
S→C {"jsonrpc":"2.0","id":6,"result":{"messages":[{"role":"user","content":{"type":"text","text":"Make a ~30-second vertical short video about: Registan at night ..."}}]}}
```

---

## 11. Claude Desktop: setup, and your test (4.3)

This is **your** step: the build session doesn't touch your app's settings. It did test the exact launch
command from a foreign folder with a minimal environment, and the server came up in 1.8 s and found the
repo, `.env` and ffmpeg.

**The snippet** (add it to your Claude Desktop config):
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
Why each part:
- **Backslashes are doubled**: that's how JSON writes `\`.
- **The space in `All codes` needs nothing**: each array item reaches uv as one argument, with no shell to split it.
- **`command` is uv's full path**: an app started from the Start menu can have a different PATH than your terminal.
- **`--directory`** makes uv run inside the repo (it finds `pyproject.toml` and `.venv` there), and
  `--env-file .env` is then found relative to it. So **your keys stay in `.env`: never put keys into this
  config file.**
- No `"env"` block is needed: images default to free placeholders.

**Steps**
1. **Finish or pause your Code sessions first.** They run inside the same Claude app, and step 4 restarts it.
2. Claude app → **Settings → Developer → Edit Config**. On your machine that file is
   `C:\Users\doniy\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\claude_desktop_config.json`.
3. Add the snippet and save. If the file already has content, put only the `"shortsmith": {...}` part inside
   the existing `"mcpServers"`, or add `"mcpServers"` next to the other top-level keys. **One wrong comma and
   Claude ignores the whole file.**
4. **Quit Claude completely** (system tray, bottom right → right-click → Quit; closing the window isn't
   enough), then start it again.
5. Check: **Settings → Developer** should list **shortsmith** as running. In a new **Chat** (not Code), the
   tools menu ("+" / "Search and tools") should show shortsmith with 5 tools.
6. Make a video. Either pick the prompt (**+ → shortsmith → Make a short video**, topic
   `cozy autumn coffee in Tashkent`, language `en`), or type:
   > Make a 30-second vertical short video about cozy autumn coffee in Tashkent, in English, with the
   > shortsmith tools. Write the script yourself and start it with create_run.

   Claude asks permission the first time each tool runs.
7. **What to expect:**
   - `create_run`, maybe rejected once (for example too few words). That's **our validator talking to Claude**,
     and it fixes the script;
   - then 6 × `generate_image` + 6 × `text_to_speech` (about 10 s);
   - then `render_video` (about 20 s);
   - then an answer with the folder. The video is `D:\All codes\shortsmith\runs\<run_id>\final.mp4`.

   **Take the screenshot for your README.** Our cost is $0; Claude's own tokens come from your Claude plan.
8. **If it fails**, read the logs in `C:\Users\doniy\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\logs\`:
   `mcp-server-shortsmith.log` (our server's stderr; it should start with the `shortsmith MCP server | root …`
   line) and `mcp.log` (Claude's side).
   - **No shortsmith in Settings → Developer:** the JSON is invalid (commas, doubled backslashes).
   - **`spawn … ENOENT`:** the uv path is wrong (`where uv` in a terminal gives the right one).
   - **`ffmpeg: NOT FOUND`** in the startup line: restart Claude so it picks up the new PATH.
9. **Known snag:** `write_script` calls Gemini's free tier, which can take a minute or more. A host may give up
   on a slow tool call. If that happens, ask Claude to write the script itself with `create_run`, which is the
   better demo anyway.

Afterwards you can attach the **`shortsmith://runs` resource** from the "+" menu to see every run. That's a
resource in action: **you** chose to load it, not the model.

---

## 12. Our agent over MCP (4.4): the numbers

**How the agent connects (`agents/mcp_tools.py`)**
- **`server_params()`** starts the same server as a **child process** of our agent (exactly what Claude
  Desktop does). The child gets only a short list of safe environment variables plus the ones it needs, passed
  by name (`MOCK_MEDIA`, `LLM_PROVIDER`, the keys). That's deliberate: **a server is someone else's code, and it
  shouldn't see all your secrets by default** (D13).
- **`chat_schema(tool, hidden=["run_id"])`** converts an MCP tool (from `tools/list`) into the Chat Completions
  format Gemini takes, and **removes `run_id`** from what the model sees.
- **`tool_caller(run_id=...)`** adds the `run_id` to every call, **last**, so a model that invents a run id
  can't point a call at another run (D11, tested). That's Phase 3's guarantee kept: the *program* picks the run.
- **`tool_message(result)`** turns an MCP result into the tool-message string. `isError` → raise
  `ToolFailed`, so Phase 3's loop handles it like any failure. `structuredContent` → the same JSON the direct
  call returned (with `"ok": true` added back).
- **Sync vs async (D10):** the MCP client is async, our loop is not. An anyio **blocking portal** runs the
  client's event loop in a background thread, and each call from the loop waits for it like a normal function.
  So **`run_loop()` didn't change at all.**
- **Resources in use (D7):** our agent's *program* (not the model) reads `shortsmith://runs/{id}/script` and
  `/status` to write the task message. That's exactly what resources are for: **context the app chooses to load.**

**The command**
```
uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855 --mcp
```

**Demo 1: the same script, direct (Phase 3) vs over MCP (Phase 4)**
| | Phase 3, direct | Phase 4, over MCP |
|---|---|---|
| Turns | 3 (12 parallel calls → render → answer) | **3, the same pattern** |
| Prompt tokens per turn | 539 / 1,313 / 1,377 | **539 / 1,313 / 1,377: identical** |
| Final video | 902 frames, 30.07 s | **902 frames**, 30.07 s, byte-identical images and subtitles |
| Wall time | 371 s | 78 s |

- **The model couldn't tell its tools had moved to another process.** The user message and all 13 tool
  messages are byte-identical between the two runs.
- **The time difference is Gemini's free tier, not MCP.** MCP's own cost was measured separately:
  **0.16 ms per call direct vs 2.41 ms over stdio**, so about 30 ms for a whole video. Starting the server
  takes ~2 s (Python start-up); the handshake itself takes ~3 ms.

**Demo 2: an injected failure through MCP** (`--mcp --fail-first text_to_speech`)
- The server raised the failure → `ToolError` → on the wire, a result with `isError: true` → our client
  raised `ToolFailed` → Phase 3's loop turned it into `{"ok": false, "error": ...}` → the model retried
  **only** the failed call → render → done. **4 turns, exactly Phase 3's behaviour, now across a process boundary.**
- The one difference: the SDK adds `Error executing tool text_to_speech: ` (37 characters) in front of the
  error, and that's **+9 prompt tokens on every later turn**. A tiny example of "every character in the
  history is paid for again on every turn".

**Total cost of Phase 4:** 7 LLM calls, $0.0035 list price, **billed $0**.

---

## 13. Errors that happened, and what they teach

**The SDK surprise: version 2.** Tutorials say `from mcp.server.fastmcp import FastMCP`. In mcp 2.3.0 that
import raises an error explaining the rename to `MCPServer`. Found by **reading the installed package before
writing code**. **Lesson:** check the version you actually have.

**E1. pytest crashed with `Windows fatal exception: access violation`** while collecting *pywin32's own test
files*. Cause: the shell's working directory had drifted into `.venv\Lib\site-packages` (from reading the
SDK's source), so pytest didn't find `pyproject.toml`, ignored `testpaths`, and collected everything. Fix:
run from the repo root. **Lesson:** pytest's config only applies when pytest starts where it can find it.

**E2. A stray `print()` leaked onto stdout, found by our own test.** This is the stdio golden rule, in practice:
- The test *"stdout carries only JSON-RPC, even when a tool prints"* failed: a line `'a stray print() inside
  a tool'` appeared on stdout.
- **But** an earlier experiment had shown prints going safely to stderr. The difference: `flush=True`.
- **Cause:** when stdout is a pipe, Python keeps `print()` output in a **memory buffer**. The SDK diverts
  stdout to stderr only *while serving*. The buffered line was written **at exit**, after the SDK had already
  put stdout back to the real channel.
- **Fix:** `sys.stdout.reconfigure(line_buffering=True)`, so every printed line is flushed immediately, while
  it still goes to stderr. A mutation check confirmed the test fails without that line.

**E3. A call got no answer at all.** The test sent all requests and closed stdin immediately. But **in stdio,
closing stdin is how the host says "shut down"**, so the server stopped and dropped the request still running.
Fix: read each answer before sending the next, and close stdin only at the end. The same rule applies to any
host: shut down only after the last answer.

**E4, E5 (small).** A shell edit broke on nested quotes (edits went through the editor instead). The SDK's
fancy logger wrapped every log line into a narrow column in log files, so it was replaced by plain one-line logging.

**Not errors, but worth knowing:**
- Argument errors now come from the SDK's validation (longer, with a Pydantic URL) instead of Phase 3's
  short wording. Errors from *our* code keep Phase 3's words.
- In SDK v2, an ordinary exception's message is **hidden** from the client (it might contain secrets).
  Only `ToolError` passes its text on, and that's why `media_tool()` converts every exception into one.

---

## 14. Tests

- **154 tests** (+37 this phase): test_mcp_server 25, test_mcp_tools 11, and +1 in test_tools.
- **The server is tested through the official SDK client, in-process**: the real handshake, real
  `tools/list`, real `tools/call`, but no subprocess, so it's fast. Voice and ffmpeg are faked as in Phase 3.
- What's covered:
  - the 5 tools and their schemas, and that descriptions match the registry docstrings;
  - `create_run`'s rules → error results with the validator's words, and no run started;
  - a media call's result byte-identical to a direct call;
  - voice that survives a "restart";
  - bad calls (unknown run, `../../etc`, scene 0, scene 9);
  - the resources and a missing run as a protocol error;
  - the prompt.
- **Two real subprocess tests:** one lists and calls tools over stdio; the other uses **raw pipes with a
  tool that prints on purpose**, and checks that every stdout line is valid JSON-RPC and the print went to
  stderr. That's the E2 test.
- **The whole Phase 3 loop through MCP** with a scripted model: it renders too early, reads the MCP error,
  fixes it, finishes. Plus `loop --from … --mcp` end to end.
- **Optional 🧠 exercise:** `exercises/phase-4/mcp_bridge.py`. Write `chat_schema()` (MCP tool → Chat
  Completions format, hiding `run_id`) and `tool_message()` (MCP result → tool-message string). It has 9 tests:
  ```
  uv run pytest exercises/phase-4
  ```

---

## Check questions (optional, answer whenever you like)
1. What does "M × N → M + N" mean, and why does it make MCP better than every app writing its own plugins?
2. What are the host, the client and the server? Which is which when Claude Desktop uses our video tools?
3. What's the difference between a tool, a resource and a prompt? Who decides to use each one?
4. What are the three kinds of JSON-RPC messages? What happens in `initialize`?
5. How does a tool error reach the model in MCP, and how is that different from a protocol error?
6. stdio vs Streamable HTTP: when would you use each? Why must a stdio server never print to stdout?
7. How do function calling and MCP fit together in our agent? Which part of the Phase 3 loop changes?
8. Why do our MCP tools take a run id instead of using a "current run"?
9. Name two security risks of MCP servers and one defence for each.

**Our build**
10. How does `media_tool()` reuse Phase 3's tools instead of copying them? Where does the input schema come from?
11. Why does every media tool take `run_id`, and why is the run's state loaded from disk on every call?
12. Why does `create_run` take the whole Script, and what happens when Claude sends a 14-word hook?
13. Why is a tool error a result (`isError`) while a missing resource is a protocol error?
14. Why did a `print()` reach stdout even though the SDK diverts stdout while serving (E2)? How was it fixed?
15. Why does our agent hide `run_id` from the model and add it to every call itself?
16. The agent over MCP used exactly the same prompt tokens as Phase 3. What does that prove? How much did MCP itself cost per call?
17. Why are your API keys in `.env` and not in the Claude Desktop config?

## Interview angles
- "MCP turns M×N custom integrations into M+N: I wrote the video tools once as an MCP server, and both
  Claude Desktop and my own agent use the same server."
- "Tools are model-controlled actions, resources are app-controlled read-only data with URIs, and
  prompts are user-chosen templates. I used all three."
- "Function calling is how a model asks its host for a tool; MCP is how the host reaches a tool that
  lives in another process. My agent loop didn't change: only its call_tool became an MCP client."
- "A stdio server's stdout is the protocol channel, so all logging goes to stderr; and since a local
  server runs with the user's permissions, mine only touches its runs folder and takes ids, not paths."
- "The same agent produced the same video over MCP with identical prompt tokens; MCP itself cost
  about 2 ms per call and 2 s to start, so the abstraction is practically free."
- "My test caught a real stdio bug: a buffered print() reached the protocol channel at shutdown. Line
  buffering fixed it, and a mutation check proved the test catches it."
- "create_run lets the host's own model write the script while my Pydantic validator enforces the
  rules, so Claude and Gemini both produce scripts that pass the same checks."
