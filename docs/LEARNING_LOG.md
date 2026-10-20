# Learning Log — Shortsmith

Claude Code appends an entry after every finished step.

## Template
### YYYY-MM-DD — Phase X, Step X.Y: <title>
- **Concepts learned:**
- **My questions → short answers:**
- **What confused me / to revisit:**
- **Interview one-liner:** (how I'd explain this step in one sentence)

---

### 2026-10-19 — Phase 0, Step 0.1: Project setup (uv, ffmpeg, repo)
- **Concepts learned:**
  - Environment management: every project gets its own isolated Python (`.venv`), so library versions never clash between projects.
  - uv = one tool for environment + install + lock + run: `uv init --package`, `uv add`, `uv run`, `uv tree`, `uv sync`.
  - `pyproject.toml` (direct deps as ranges) → `uv.lock` (every package pinned exactly, with sha256 hashes) → `.venv/` (installed packages, never committed). Same trio as `package.json` / `package-lock.json` / `node_modules/` in JS.
  - Direct vs transitive dependencies; dependency resolution (one version per package that satisfies everyone).
  - src layout (`src/shortsmith/`) and `[project.scripts]`, which creates the `shortsmith` command.
  - Secrets: `.env` is gitignored, `.env.example` is committed with key names only. `.gitignore` wildcards, `!` exceptions, debugging with `git check-ignore -v <file>`.
  - Warning "Failed to hardlink files": uv's cache is on C:, the project on D:, and hardlinks can't cross drives, so uv copies instead. Harmless.
- **My questions → short answers:**
  - Why did 1 package (`openai`) install 15? → openai needs its own packages, which need theirs (transitive dependencies). `uv.lock` records the exact version of all of them.
  - Why commit `uv.lock` but not `.venv/`? → the lock is a small recipe that reproduces the exact environment with `uv sync`; `.venv` is big (19 MB / 2,370 files for one library), machine-specific (Windows paths, .exe files) and rebuildable.
  - What breaks without `!.env.example`? → `.env.*` matches it too, so the template is silently never committed and a cloner doesn't know which keys to set.
  - What is uv? → a fast Rust tool that replaces venv + pip + pip freeze (+ activation) for a Python project.
  - What is a transitive dependency? → a dependency of your dependency (`shortsmith → openai → httpx2 → httpcore2 → h11`). Rule: if you import it, declare it (we'll `uv add pydantic` in 1.4 even though it's already installed).
  - What is dependency resolution? → finding one version of each package that satisfies all version ranges at once, or failing with "No solution found".
  - What is typing-extensions? → backports new `typing` features (type hints) to older Pythons; libraries like pydantic depend on it. Pydantic reads type hints at runtime to validate data.
  - What is pyproject.toml? → the standard config file of a Python project: metadata, Python version, dependencies, commands, build system, tool settings.
  - Does `.env.*` cover `.env.local`, `.env.prod`…? → yes, every `.env.<something>`, but not plain `.env` (hence its own line). `!` only means "commit this"; git never checks content, so `.env.example` stays safe only if it holds names, never values. `.gitignore` doesn't affect files git already tracks.
- **What confused me / to revisit:** `uv.lock` and `!` negation in `.gitignore` were new. Revisit "if you import it, declare it" when adding pydantic in Step 1.4.
- **Interview one-liner:** "I manage the project with uv: pyproject.toml declares direct dependencies as ranges, uv.lock pins every transitive dependency with hashes so `uv sync` reproduces the exact environment anywhere, and secrets live in a gitignored .env with a committed names-only template."
- **Decisions this step:** adopted the new PLAN.md (LangGraph + MLflow in Phase 6). Project order: Shortsmith first, Uzbek text-to-SQL fine-tuning second. Target role: Higgsfield AI Engineer (Almaty, on-site).

---

### 2026-10-19 — Phase 0, Step 0.2: API keys (OpenAI, fal.ai, YouTube)
- **Concepts learned:**
  - An API key is a password that tells a service which account to bill. Three rules: least privilege (one key per project, restricted to what it needs), limits (prepaid credits, auto-recharge off, budget alerts), rotation (revoke + recreate when leaked or expired).
  - Environment variables: values the OS hands to a running program. `.env` is just a file that tools load into them; `uv run --env-file .env` loads it for one run only.
  - OpenAI: separate `shortsmith` project in the existing org, data sharing on (free daily tokens, but prompts may be used for training, so no private data), $5 project budget, key `shortsmith-dev` (owned by me, 180-day expiry).
  - fal.ai: serverless GPU inference for image/video models, billed per output. Queue API: submit → wait (polling / subscribe / webhook) → result URL. API-scoped key, no credits until Phase 2.1; `MOCK_MEDIA=1` keeps debugging free.
  - YouTube Data API: real trend data for the Researcher (Phase 5) so scripts are grounded in evidence, not the model's frozen knowledge. Search costs 100 of 10,000 free daily units. No Shorts-only filter (`videoDuration=short` = under 4 min).
  - Verifying keys for free: OpenAI `models.list` (0 tokens), YouTube `i18nRegions` (1 unit), fal only by shape (`id:secret`). Never print any part of a key; send Google keys in the `X-Goog-Api-Key` header, not the URL.
- **My questions → short answers:**
  - Why do we need the YouTube API? → the Researcher needs what is getting views *now*; an LLM would invent trends. It's the only free, official, stable trend source.
  - What is fal.ai? → one API key for many image/video models running on fal's GPUs, pay per output (~$0.003 per FLUX schnell image).
  - Will the $50 Google card hold come back? → yes, it's a reservation, not a charge; the bank releases it, usually within a week (up to 7–10 business days). A som card may lose a little to exchange rates if the bank debits and refunds instead.
  - Why a separate OpenAI project? → revoke Shortsmith's key without breaking the Yukdosh bot, and see/limit Shortsmith's cost separately. Credits, org rate limits and (I believe) the free token pool are still shared.
  - Why is `API_KEY_SERVICE_BLOCKED` good? → proof the YouTube key is restricted: if it leaks, the worst case is a used-up free quota, not paid Google APIs billed to my card.
  - Is deleting a pushed `.env` in the next commit enough? → no, git history keeps it. Rotate every key first (assume it was copied), then clean up (`git rm --cached`, optional history rewrite). Making a repo public later exposes its whole history.
- **Errors that were lessons:**
  - `uv run --env-file "D:\All codes\shortsmith\.env"` → "No environment file found at: codesshortsmith.env": uv split the path at the space and treated `\` as an escape character. Fix: run from the repo root with `--env-file .env`.
  - Key "pasted" but missing: Windows 11 Notepad doesn't ask to save on close. Always Ctrl+S; the check script caught it.
  - Google Cloud kept the old "Instagram" project selected after creating `shortsmith`: always check the project in the top bar before enabling APIs or creating keys.
  - Google Cloud now requires 2-step verification; turned it on, saved backup codes offline.
- **What confused me / to revisit:** all three check questions were new (project separation, least privilege, git history). Revisit before making the repo public in Phase 2.
- **Interview one-liner:** "Each provider gets its own least-privilege key per project, with budgets and expiry, loaded from a gitignored .env at runtime; I verify keys with free calls that never print secrets, and I know a leaked key means rotate first, clean git history second."
- **Decisions this step:** set up all three keys now instead of just-in-time. Declined to build on the Google Cloud $300 trial: it's one-time, the 90-day clock starts on activation, and free trials can't use GPUs. Reminder: OpenAI key `shortsmith-dev` expires around 2027-04-17.

---

### 2026-10-20 — Phase 1, Step 1.1: First LLM call + usage (code done, live run pending)
- **Concepts learned:**
  - A Chat Completions call = a list of messages with roles (`system` = rules, `user` = request, `assistant` = earlier replies) → one reply + a `usage` report.
  - The model is stateless: a "conversation" is the code resending the whole history every call.
  - Tokens: `prompt_tokens` (input) vs `completion_tokens` (output, pricier); `cached_tokens` = repeated prompt beginnings (1,024+ tokens) billed cheaper.
  - Chat Completions (portable "OpenAI-compatible" format: Groq, Ollama, vLLM…) vs Responses API (newer, built-in tools and server-side state, OpenAI-only).
  - `llm.py`: `ChatResult` Pydantic model (exact model version, tokens, latency; `total_tokens` as a computed property), `@cache` lazy client (import needs no key, one reused connection), `time.perf_counter()` for timing, `content or ""` because content is `None` on refusals/tool calls, no error handling yet on purpose (loud errors; the SDK already retries 429/5xx twice).
  - Pydantic = data checked at the border: converts what it safely can (`"12"` → 12), rejects the rest with clear errors, dumps to JSON.
  - `uv add pydantic` changed pyproject (our wish, a range) and uv.lock (shortsmith now depends on it directly); nothing was downloaded since openai already needed it.
- **My questions → short answers:**
  - What was Pydantic? → a library that validates data against a shape declared with type hints; it guards every LLM output and every object passed between agents.
  - pyproject.toml vs uv.lock again? → shopping list (ranges, written by me / `uv add`) vs receipt (exact versions + hashes of every package, written only by uv). `.venv` is the fridge.
  - Stop Yukdosh and wait for the free-token reset? → pausing it does save free tokens if they work, but the logs showed the resets at 00:00 UTC on 10-19 and 10-20 didn't help while the balance was $0. Adding credits is the only reliable fix.
- **Check questions (skipped; answers for revision):**
  - What is sent on the 3rd call? → the whole history: system + user1 + assistant1 + user2 + assistant2 + user3. `prompt_tokens` grows every turn, so long chats get expensive and eventually hit the context window; the app must trim or summarize.
  - Why the lazy `@cache` client? → importing `llm.py` needs no API key (offline tests and mocks don't crash), and one client is reused (no new connection per call).
  - Where was the real 429 reason? → in the error body (`code: credit_balance_exhausted`, `type: insufficient_quota`), not the status number or class name. The Step 0.2 check used the free `models.list`, which works at $0: it proved the key is valid, not that it can spend.
- **Errors that were lessons:** first real call → `RateLimitError 429 credit_balance_exhausted`: the OpenAI org ("Personal") is at $0. Same org as Yukdosh, whose parser has failed since 2026-10-18 15:46 UTC; the Yukdosh server is stopped on purpose until credits are added. Read tracebacks bottom-up: the last line is the error, the lines above are the path through my code, then the library.
- **What confused me / to revisit:** run the live call and read a real `usage` table once credits exist.
- **Interview one-liner:** "Every LLM call goes through one thin wrapper that returns a validated Pydantic result with token usage and latency, so cost tracking, retries and model choice live in one place."

---
