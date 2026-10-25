# CLAUDE.md — Shortsmith (learning project)

## Who I am
I'm Doniyor, a 4th-year AI student and data analyst (Python/pandas, SQL, ClickHouse, FastAPI, BI tools).
I'm building this project for two reasons, and both matter equally:
1. **To learn** LLMs, agents, tool use, MCP, RAG and AI-native workflows deeply, well enough to explain every line in an interview.
2. **To ship** a working portfolio demo for AI-engineering job applications.

**Target role:** Higgsfield AI Engineer (Almaty, on-site); its requirements match the table in PLAN.md §1.
Then AI/ML roles in Uzbekistan. Project order: Shortsmith (this repo) first, then an Uzbek
text-to-SQL fine-tuning project. Plan to apply after Phase 4 (end-to-end video + MCP demo).

The full spec is in `docs/PLAN.md`. Read it at the start of every session.

## How you must work with me (learning workflow, owner's choice 2026-10-21)

Learning must never wait on building. Per-step "go" pauses, per-step commits and long test logs in
chat interrupted my flow, so each phase now runs like this:

1. **Two sessions per phase.**
   - **Build session** (background, started from the teaching session): builds *every* step of the
     phase, runs tests and demos, fixes its own errors, verifies quietly, and commits once at the end
     of the phase. It owns `src/`, `tests/`, `exercises/`, `pyproject.toml`, `uv.lock`. It writes
     `docs/lessons/phase-N-build-notes.md`: what was built, results (numbers), every error hit and how
     it was found and fixed, decisions and their reasons.
   - **Teaching session**: teaches while the build runs. It owns `docs/lessons/phase-N.md`,
     `docs/LEARNING_LOG.md` and this file. The two sessions never edit the same files.
2. **One detailed lesson per phase** (`docs/lessons/phase-N.md`), written from the *finished* code
   and the build notes, with nothing skipped: every concept, every file and function and *why* it's
   written that way, the errors that happened (errors are lessons), costs, and interview angles.
   I read it at my own pace and ask questions; answer them fully, even off-topic ones.
3. **Interactive explainers only for the important, hard parts** (e.g. ffmpeg filter graphs, the agent
   loop, MCP, embeddings/RAG, LangGraph state): visual, self-paced, with small quizzes.
4. **🧠 steps become exercises off the critical path.** The build session writes the real code, then
   puts an exercise copy in `exercises/phase-N/` (same function, body removed, plus its tests). I do
   them whenever I want (`uv run pytest exercises/phase-N`); they never block progress. Review my
   exercise code when I ask, and give the full solution only if I ask for it.
5. **Check questions** go at the end of each lesson: optional, answered whenever I like; then tell me
   honestly what I got right or wrong.
6. **Keep a learning log**: one entry per phase in `docs/LEARNING_LOG.md` (concepts, my questions and
   short answers, errors, decisions, interview one-liner).
7. **Say exactly where things go.** For every command, say where to run it (e.g. "terminal, repo root").
8. **Be honest.** If my idea or code is wrong or overcomplicated, say so and explain why.

## Cost safety (important: I'm paying)
- Never run paid generation (images, video) in a loop or batch without asking me first, and tell me the estimated cost.
- Default to `MOCK_MEDIA=1` (placeholder images/clips) while building and debugging the pipeline. Use real APIs only when I ask.
- Every LLM and media call must go through the cost/token ledger (see PLAN.md, Phase 1).
- Never commit `.env` or API keys.

## Tech conventions
- Python 3.11+, managed with `uv`. Source code in `src/shortsmith/`.
- Pydantic models for all data passed between agents. No loose dicts.
- Type hints everywhere. Short docstrings that explain *why*, not *what*.
- `pytest` tests for pure logic (parsing, ffmpeg command building, ledger). Mock the LLM in tests.
- Frontend: Next.js + TypeScript in `web/`.
- Commit at the end of a phase, or when it's genuinely needed (before a risky change, or when I ask).
  Don't commit or ask about committing after every step: it interrupts my learning flow (owner's request,
  2026-10-21). Keep each commit message clear about which steps it covers.

## Gotchas on this machine
- The repo path has a space (`D:\All codes\shortsmith`). Run uv from the repo root with relative paths
  (`uv run --env-file .env ...`). An absolute `--env-file` path gets split at the space.
- Never open, print or echo `.env`. Check keys only with free calls that print OK/error, never key fragments.
- Keys: OpenAI project `shortsmith` (key `shortsmith-dev`, expires ~2027-04-17, rotate before then);
  fal.ai key with API scope (no credits until Phase 2.1); YouTube key restricted to YouTube Data API v3
  in Google Cloud project `shortsmith`.
- **LLM provider (owner's decision, 2026-10-21): build and test on Gemini's free tier** (`LLM_PROVIDER=gemini`
  in `.env`, OpenAI-compatible endpoint, default model `gemini-3.5-flash-lite`; `gemini-3.8-flash` for
  quality-sensitive steps). Gemini key lives in AI Studio project `gen-lang-client-0252045076`, Free tier:
  never click "Set up billing". Limits are per-minute/per-day requests (see aistudio.google.com/rate-limit).
- **Keep the OpenAI path intact** (SDK, `PROVIDERS["openai"]`, key `shortsmith-dev`, prices). The OpenAI org
  ("Personal", shared with the owner's production project Yukdosh) is at $0 since 2026-10-18
  (`429 credit_balance_exhausted`). Switch back later with `LLM_PROVIDER=openai` after adding credits.
- **Video target (owner's choice, 2026-10-21): 6 images + 1 hero clip per video** (~$0.27–0.62, so $5 ≈ 8–18
  videos). Hero clips are always approval-gated and never made while `MOCK_MEDIA=1`; check the live fal price
  before the first one (the old plan figure of $0.10–0.25 per clip was too optimistic).
- Token counts are model-specific: tiktoken is exact only for OpenAI models. The ledger always records the
  provider's own `usage`.
- Gemini 3 tool calling: append the assistant message exactly as returned (`ChatResult.message`); it carries
  `extra_content.google.thought_signature`, and a history without it is rejected with a 400. Free-tier LLM
  turns can take 13–250 s (server-side queueing), so live agent demos are slow; that's not a bug.
- Free tier, measured in Phase 6: **each Flash model allows only 20 requests a day** (resets midnight Pacific);
  flash-lite has much more. Free-tier embeddings: 100 texts per minute. The Critic's model is
  `SHORTSMITH_CRITIC_MODEL` (default `gemini-3.8-flash`, fallback `gemini-3.6-flash`).
- Gemini Flash models think, and the compatible endpoint counts thinking only in `total_tokens`; `_complete()`
  prices `total - prompt - completion` as reasoning (output rate). Keep that when touching the ledger.
- Windows Smart App Control blocks brand-new native wheels (e.g. uuid-utils 0.18, pandas 3.1, SQLAlchemy 2.1):
  `[tool.uv] constraint-dependencies` pins versions that load. Never change the security setting.
- The owner's terminal is Windows PowerShell 5.1 (console code page IBM437): `>` and `|` re-encode a program's output,
  so Uzbek text gets garbled. Use `-o FILE` or `--json` (ASCII) on list/show/cost, or `cmd /c "... > file"`.

## Current progress
Update this line at the end of every step:
**Current phase/step:** Phase 8 next (web app). Phase 7 built 2026-10-24: the `shortsmith` command
(`cli.py`, `console.py`, `views.py`, `runs.py`, `costs.py`, `budget.py`, `rerun.py`): make / resume / rerun-scene /
list / show / cost / doctor / mcp, `--budget` (list price, exit 4), guarded `--real`, `--json`; commit 04cfd7e (code) +
the lesson commit, lesson `docs/lessons/phase-7.md`. 391 tests.
Make a video now with `uv run shortsmith make "topic"` (the command loads `.env` itself; `python -m shortsmith.make`
still works). `uv run shortsmith doctor` checks the setup without printing keys.
Phase 6 (LangGraph + MLflow) done 2026-10-24. Phase 5 (research + RAG) done 2026-10-23. Owner still to do: the Claude Desktop test + README screenshot (phase-4
lesson §11; config file is under `%LOCALAPPDATA%\Claude-Data\...`, opened via Settings → Developer).
Exercises waiting (not done yet): `exercises/phase-2/` (Ken Burns), `exercises/phase-3/` (the agent loop),
`exercises/phase-4/` (optional: MCP bridge), `exercises/phase-5/` (RAG math), `exercises/phase-6/` (graph runner), `exercises/phase-7/` (optional: budget rules).
Real FLUX images wait for credits or a free image source.
Open items: Step 0.3 deferred by choice · OpenAI live comparison (tiktoken predicted 35 prompt_tokens;
Gemini reported 26) waits for OpenAI credits.
