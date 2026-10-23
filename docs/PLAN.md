# Shortsmith — Project Plan

> An AI agent system that turns **one line of text** (a topic or brand) into a **finished ~30-second vertical short video**: it researches trends, writes the script, storyboards, generates visuals and voice, edits, and critiques its own work.

```
$ shortsmith make "coffee shop in Tashkent, cozy autumn vibe" --lang uz
→ researching trends... writing 3 scripts... picked #2 (hook score 8.4)
→ storyboarding 6 scenes... generating images... voiceover... editing...
→ critic: scene 4 weak (off-topic) → regenerating
✔ runs/2026-10-20_1532/final.mp4   (cost: $0.03, tokens: 48k)
```

---

## 1. Why this project (job-requirement mapping)

| Required skill | Where it's demonstrated |
|---|---|
| Python + JavaScript | Python agents/backend; Next.js (TypeScript) frontend |
| Prototypes, automations, internal tools | CLI tool, daily automated run → Telegram, web dashboard |
| LLMs | Every agent; structured outputs; model choice per task |
| Tokens / context windows | Token ledger, per-step budgets, context trimming between agents |
| APIs & CLI tooling | OpenAI, fal.ai, YouTube Data API, edge-tts, ffmpeg; Typer CLI |
| MCP ecosystems | Own MCP server exposing the media tools; used from Claude Desktop **and** from our own agents |
| Agent frameworks | Hand-built agent loop first (Phase 3) to understand the mechanics, then **LangGraph** as the real orchestrator (Phase 6) |
| Tool use & orchestration | Function calling; LangGraph multi-agent graph with a critic/retry loop and checkpointing |
| RAG / retrieval | Vector store of viral hooks + brand guidelines used to ground scripts |
| LLMOps | **MLflow** tracing of every run + prompt-version evaluation by Critic score |
| AI-native coding workflow | Built with Claude Code (CLAUDE.md, plan mode, learning log) and described in the README |

---

## 2. Architecture

```
            ┌──────────────────┐
 topic ───► │ LangGraph graph  │  (RunState = graph state, checkpointed after every node;
            └──────┬───────────┘   every run traced in MLflow)
   ┌───────────┬───┴──────┬────────────┬────────────┬──────────┐
   ▼           ▼          ▼            ▼            ▼          ▼
Researcher   Writer    Director     Producer      Editor     Critic
(trends +    (3 script (scenes +    (calls media  (ffmpeg    (reviews frames
 RAG hooks)  options,  visual       tools via     assembly)  + script, sends
             picks 1)  prompts)     MCP)                     weak scenes back)
                                       │
                              ┌────────┴─────────┐
                              │   MCP server     │  tools: search_trends, retrieve_hooks,
                              │ (shortsmith-mcp) │  generate_image, animate_image,
                              └──────────────────┘  text_to_speech, render_video
```

**Key design ideas (things to learn and explain in interviews):**
- **Agents are just LLM calls with a role, a context and tools.** No magic.
- **Structured outputs:** every agent returns a validated Pydantic object, never free text.
- **Context engineering:** each agent gets only what it needs (e.g. the Editor never sees research notes), which saves tokens and reduces confusion.
- **Evaluator–optimizer loop:** the Critic scores the result, and only failing scenes are regenerated, up to N times. In LangGraph this is a *conditional edge* from Critic back to Producer.
- **Durability:** LangGraph checkpoints state after each node, so a crashed run resumes instead of paying for everything again.
- **Observability:** every run is an MLflow trace (each node's prompt, output, tokens, latency), so quality and cost are measured, not guessed.
- **Cheap by default:** images + ffmpeg motion; a real AI video clip for **1 "hero" scene per video** (usually the hook), always after an approval step.

---

## 3. Tech stack & cost

| Part | Tool | Cost |
|---|---|---|
| LLM | Gemini free tier (`gemini-3.5-flash-lite`, `gemini-3.8-flash`) through the OpenAI-compatible endpoint while building; OpenAI `gpt-4.1-mini` kept as a switch (`LLM_PROVIDER`) | ~$0 |
| Embeddings | Gemini `gemini-embedding-2` at 768 dimensions (free tier; OpenAI's model stays behind `LLM_PROVIDER=openai`) | $0 on the free tier |
| Vector DB | Chroma (local) | free |
| Images | fal.ai FLUX schnell (~$0.003/image) | ~$0.02 per video |
| Hero clip (1 per video) | fal.ai image-to-video: Wan 2.5 480p / Veo 3.1 Lite from ~$0.05/s, Kling ~$0.07–0.11/s, premium models $0.30–0.75/s (checked 2026-10-21; verify the live price first) | ~$0.25–0.60 per 5-s clip |
| Voice | `edge-tts` (free; Uzbek voices `uz-UZ-MadinaNeural`, `uz-UZ-SardorNeural`) | free |
| Subtitles | edge-tts word-boundary timings → `.srt` | free |
| Orchestration | LangGraph (standalone, **without** LangChain) + SQLite checkpointer | free |
| LLMOps | MLflow 3 tracing (`mlflow.openai.autolog()`) + local MLflow UI | free |
| Editing | ffmpeg (subprocess) | free |
| Trends | YouTube Data API v3 (free quota: 10k units/day; one search = 100 units) | free |
| Agent protocol | `mcp` Python SDK (FastMCP) | free |
| CLI / API | Typer / FastAPI + Server-Sent Events | free |
| Frontend | Next.js + TypeScript → Vercel | free |
| Backend hosting | Azure App Service / small VM ($200 credit) | credits |
| Notifications | Telegram bot | free |

**Cost per 30-second video:** images only ≈ $0.02 ($5 ≈ 270 videos). **Target setup, 6 images + 1 hero clip ≈ $0.27–0.62 ($5 ≈ 8–18 videos).** Every scene as a clip ≈ $1.50–3.60 ($5 ≈ 1–3 videos).

**Total budget target:** under $15 for the whole project.

**Data-sharing note:** with data sharing on, OpenAI can use prompts for training, so never send private or company data.

---

## 4. Repo structure (target)

```
shortsmith/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── PLAN.md
│   └── LEARNING_LOG.md
├── pyproject.toml
├── .env.example            # OPENAI_API_KEY, FAL_KEY, YOUTUBE_API_KEY, TELEGRAM_*, MOCK_MEDIA
├── data/
│   └── hooks/              # seed library of viral hooks & brand guidelines (markdown/json)
├── src/shortsmith/
│   ├── config.py           # pydantic-settings
│   ├── models.py           # Script, Scene, RunState, CriticReport ...
│   ├── llm.py              # OpenAI wrapper + token/cost ledger + structured output helper
│   ├── ledger.py
│   ├── media/
│   │   ├── images.py       # fal FLUX (+ mock)
│   │   ├── animate.py      # image-to-video (+ mock)
│   │   ├── tts.py          # edge-tts voiceover + word timings → .srt
│   │   └── ffmpeg.py       # ken burns, concat, subtitles, music
│   ├── research/
│   │   ├── youtube.py
│   │   └── rag.py          # ingest + retrieve (Chroma)
│   ├── agents/
│   │   ├── loop.py         # hand-built tool-calling loop
│   │   ├── researcher.py  writer.py  director.py
│   │   ├── producer.py    editor.py  critic.py
│   │   └── graph.py        # LangGraph: nodes, edges, critic loop, checkpointer
│   ├── tracing.py          # MLflow setup + custom spans/metrics
│   ├── mcp_server.py
│   ├── cli.py              # Typer
│   └── api.py              # FastAPI + SSE
├── web/                    # Next.js app
├── runs/                   # one folder per run: state.json, scenes/, final.mp4 (gitignored)
└── tests/
```

---

## 5. Core data models (sketch)

```python
class Scene(BaseModel):
    idx: int
    narration: str            # what the voice says
    visual_prompt: str        # prompt for the image model
    duration_s: float
    motion: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "ai_clip"]
    image_path: str | None = None
    clip_path: str | None = None

class Script(BaseModel):
    title: str
    hook: str                 # first 2 seconds, the most important line
    scenes: list[Scene]
    cta: str
    language: Literal["en", "uz", "ru"]

class CriticReport(BaseModel):
    overall_score: float      # 0–10
    weak_scenes: list[int]
    reasons: dict[int, str]

class RunState(BaseModel):
    run_id: str
    topic: str
    research_notes: str | None = None
    script_options: list[Script] = []
    chosen: Script | None = None
    audio_path: str | None = None
    final_path: str | None = None
    critic_rounds: list[CriticReport] = []
    tokens_used: int = 0
    cost_usd: float = 0.0
```

---

## 6. Phases (learning path)

Each phase has: **Goal → Steps → Concepts to learn → Check questions → Done when.**
🧠 = "YOUR TURN": I write the core part myself, and Claude gives hints and reviews.

### Phase 0 — Setup & AI-native workflow
**Goal:** a clean repo and understanding of how Claude Code works.
- 0.1 Install `uv`, `ffmpeg`, Node.js; create the repo, `pyproject.toml`, `.env.example`, `.gitignore`.
- 0.2 Get API keys: OpenAI (enable data sharing, add $5), fal.ai, YouTube Data API.
- 0.3 Learn the Claude Code basics: CLAUDE.md, plan mode, how to review diffs, `/clear`, when to start new sessions.
**Concepts:** environment management, secrets, AI-native workflow.
**Check:** Why is `.env` never committed? What does CLAUDE.md do and when is it read?
**Done when:** `uv run python -c "import openai"` works and `ffmpeg -version` prints.

### Phase 1 — LLM fundamentals
**Goal:** call an LLM properly and *understand tokens*.
- 1.1 `llm.py`: a basic chat call; print `usage` (prompt vs completion tokens).
- 1.2 Use `tiktoken` to count tokens yourself; compare with the API's count; explore what a context window is and what happens when you exceed it.
- 1.3 🧠 `ledger.py`: record every call (step, model, tokens in/out, cost) to JSON.
- 1.4 Structured outputs: `topic → Script` as a validated Pydantic object.
- 1.5 Experiment: temperature 0 vs 1 on the same prompt; system vs user prompt roles.
**Concepts:** tokens, tokenization, context window, input vs output pricing, temperature, system prompts, structured outputs / JSON schema.
**Check:** Why do output tokens cost more? What happens to old messages when context fills up? Why are structured outputs better than parsing text?
**Done when:** `uv run python -m shortsmith.llm "topic"` prints a valid Script plus its token cost.

### Phase 2 — Media primitives (first video!)
**Goal:** turn a *hardcoded* Script into an mp4, with no agents yet.
- 2.1 `images.py`: FLUX schnell via `fal-client` (vertical 9:16) + `MOCK_MEDIA` placeholder mode.
- 2.2 `tts.py`: edge-tts voiceover (en + uz) and its word-boundary timings → `.srt` subtitle file.
- 2.3 🧠 `ffmpeg.py`: build the Ken Burns zoom/pan command for one image.
- 2.4 Concatenate scenes, burn in subtitles, mix background music → `final.mp4`.
- 2.5 Tests for the ffmpeg command builder and the `.srt` writer.
**Concepts:** REST/async APIs, polling vs subscribe, ffmpeg filter graphs (`zoompan`, `concat`, `subtitles`), audio/video sync.
**Check:** Why generate images at 9:16? How do subtitle timings stay in sync with the voice?
**Done when:** a first real 30-second video is created for about $0.02.

### Phase 3 — Tool use from scratch
**Goal:** understand what "an agent" actually is by building the loop by hand.
- 3.1 Define tools as JSON schemas (`generate_image`, `text_to_speech`, `render_video`).
- 3.2 🧠 `agents/loop.py`: the loop — send messages + tools → model returns tool calls → execute → append results → repeat until done or `max_steps`.
- 3.3 Error handling: a tool fails → return the error to the model → it retries or adapts.
- 3.4 Add step and token budgets; log every turn.
**Concepts:** function calling, tool schemas, the agent loop, stop conditions, tool errors as feedback, parallel tool calls.
**Check:** Who executes the tool, the model or your code? Why do you need `max_steps`? What's in the message history after 3 tool calls?
**Done when:** given a script, the agent decides on its own which tools to call and produces the video.

### Phase 4 — MCP
**Goal:** expose our tools as a standard MCP server usable by *any* MCP client.
- 4.1 Learn the MCP basics: host / client / server; tools vs resources vs prompts; stdio vs HTTP transport.
- 4.2 `mcp_server.py` with FastMCP: move the media tools in.
- 4.3 Connect it to **Claude Desktop** and generate a video by chatting (screenshot this for the README!).
- 4.4 Make our own agent loop use the tools through an MCP client instead of direct imports.
**Concepts:** MCP protocol, why a standard protocol matters (M×N → M+N), transports, tool discovery.
**Check:** What's the difference between a tool and a resource? Why is MCP better than every app writing its own plugins?
**Done when:** the same server works from both Claude Desktop and our agent.

### Phase 5 — Research + RAG
**Goal:** scripts grounded in real trends and proven hook patterns.
- 5.1 `youtube.py`: fetch trending/top Shorts for a topic (titles, view counts); watch the API quota.
- 5.2 Build `data/hooks/`: ~50–100 viral hook patterns + 2 sample brand guidelines.
- 5.3 🧠 `rag.py`: chunk → embed → store in Chroma → retrieve top-k for a query.
- 5.4 The Researcher agent combines trends + retrieved hooks into short research notes.
- 5.5 Experiment: script quality with RAG vs without (simple side-by-side).
**Concepts:** embeddings, cosine similarity, chunking strategies, top-k, grounding, RAG vs fine-tuning vs long context.
**Check:** Why not just put all 100 hooks in the prompt? What happens if chunks are too big or too small?
**Done when:** the Writer's scripts visibly use retrieved hook patterns.

### Phase 6 — Multi-agent orchestration with LangGraph + MLflow
**Goal:** the full pipeline, end to end, built the way companies build agents.
- 6.1 LangGraph basics on a toy graph (3 nodes, 1 conditional edge): state, nodes, edges, `compile()`, `invoke()`. LangGraph is used **standalone**, without LangChain.
- 6.2 🧠 `graph.py`: nodes Researcher → Writer (3 options + self-pick) → Director → Producer → Editor → Critic, with `RunState` as the graph state.
- 6.3 Critic node: review 3–4 sampled frames (vision) + the script → `CriticReport`.
- 6.4 🧠 Conditional edge = the evaluator–optimizer loop: weak scenes → back to Producer (only those scenes), max 2 rounds → else END.
- 6.5 Checkpointing with the SQLite checkpointer: kill a run halfway, then resume it without paying again.
- 6.6 Human-in-the-loop: an `interrupt` before expensive hero-clip generation ("this will cost ~$0.30, continue?").
- 6.7 `tracing.py`: MLflow tracing (`mlflow.openai.autolog()` + a span per node); open the MLflow UI and read one full run trace.
- 6.8 Evaluation: run 5 topics with prompt v1 vs v2, log Critic scores + cost to MLflow, compare.
- 6.9 Context engineering: trim what each node receives; compare token use before and after in MLflow.
**Concepts:** graph-based orchestration, state reducers, conditional edges, checkpointing & durable execution, human-in-the-loop, tracing vs logging, LLM evaluation, prompt versioning, hand-built loop (Phase 3) vs framework trade-offs.
**Check:** What does LangGraph give you that your Phase 3 loop didn't? Why checkpoint after every node? What's the difference between a trace and a log? How do you know prompt v2 is actually better?
**Done when:** `topic → final.mp4` works with a critic loop, resume, an approval step, and every run visible in MLflow.

### Phase 7 — CLI tool
- 7.1 Typer commands: `make`, `list`, `show <run>`, `rerun-scene <run> <idx>`, `cost`.
- 7.2 Flags: `--lang`, `--hero-clips N`, `--budget`, `--mock`.
- 7.3 Rich progress output (spinners, cost per step).
**Done when:** a person can use the tool without reading the code.

### Phase 8 — Web app (JavaScript)
- 8.1 `api.py`: FastAPI endpoints to start a run + an SSE stream of progress events.
- 8.2 Next.js UI: topic input → live pipeline timeline → scene thumbnails appear as they're generated → final video player.
- 8.3 Show tokens and cost per step in the UI.
**Concepts:** SSE vs WebSockets, async jobs, React state, TypeScript basics.
**Done when:** the full run is visible live in the browser.

### Phase 9 — Automation & deploy
- 9.1 A daily scheduled run: "trend of the day" → video → sent to a Telegram bot.
- 9.2 Deploy the backend (Azure) and frontend (Vercel); connect the domain.
- 9.3 Rate limits and a spend cap on the public demo.
**Done when:** a live link works and the daily video arrives in Telegram.

### Phase 10 — Portfolio packaging
- 10.1 README: demo GIF, architecture diagram, design decisions, cost table, "how I built it with Claude Code".
- 10.2 A 2-minute demo video (CLI + web + Claude Desktop MCP).
- 10.3 Interview prep: be able to explain every phase's "Check" questions without notes.

---

## 7. Scope rules
- **Core = Phases 0–6.** That alone is a strong project. Phases 7–10 are polish.
- If something takes too long, simplify. A working, smaller demo beats a perfect unfinished one.
- Quality focus: the **hook** (first 2 seconds) and **visual consistency** across scenes (same style suffix in every image prompt).
