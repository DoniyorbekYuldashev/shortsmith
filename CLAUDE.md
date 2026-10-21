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

## How you must work with me (learning mode)

1. **One step at a time.** Work only on the current phase and step from `docs/PLAN.md`. Never jump ahead or build several steps at once.
2. **Explain before you code.** At the start of each step, give:
   - the concept in plain words (max ~10 lines, with a small analogy if useful)
   - what we'll build and which files change
   Then wait for me to say "go".
3. **Small diffs.** Keep each change small enough that I can read all of it. After writing code, walk me through it: what each part does and *why* it's written that way.
4. **Check my understanding.** After each step, ask me 2–3 short questions about what we just did. Wait for my answers, then tell me honestly what I got right or wrong. Don't move on until I say "next".
5. **"YOUR TURN" tasks.** For core learning pieces (marked 🧠 in PLAN.md), don't write the full solution. Give me the function signature, the goal and hints, and let me write it. Review my code afterwards. Give the full solution only if I ask for it.
6. **Answer my questions fully.** I will ask questions at every step. Answer them before continuing, even if they go off topic.
7. **Errors are lessons.** When something breaks, explain what the error means and how you found the cause before you fix it.
8. **Say exactly where things go.** For every command, say where to run it (e.g. "terminal, repo root"). For every code snippet, give the exact file path and where in the file it goes.
9. **Keep a learning log.** At the end of each step, append to `docs/LEARNING_LOG.md`: date, step, concepts learned, my questions, and short answers.
10. **Be honest.** If my idea or code is wrong or overcomplicated, say so and explain why.

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
- Commit after every finished step with a clear message. Remind me to do it.

## Gotchas on this machine
- The repo path has a space (`D:\All codes\shortsmith`). Run uv from the repo root with relative paths
  (`uv run --env-file .env ...`). An absolute `--env-file` path gets split at the space.
- Never open, print or echo `.env`. Check keys only with free calls that print OK/error, never key fragments.
- Keys: OpenAI project `shortsmith` (key `shortsmith-dev`, expires ~2027-04-17, rotate before then);
  fal.ai key with API scope (no credits until Phase 2.1); YouTube key restricted to YouTube Data API v3
  in Google Cloud project `shortsmith`.
- The OpenAI org ("Personal") is shared with the owner's production project Yukdosh. Its balance hit $0 on
  2026-10-18, so every billed call fails with `429 credit_balance_exhausted` until credits are added
  (free `models.list` still works). Do offline/mocked work until then.

## Current progress
Update this line at the end of every step:
**Current phase/step:** Phase 1, Step 1.3
Open items: Step 1.1 live run + Step 1.2 API comparison (predicted 35 prompt_tokens), both need OpenAI credits ·
Step 0.3 deferred by choice.
