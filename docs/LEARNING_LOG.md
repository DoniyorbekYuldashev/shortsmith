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
