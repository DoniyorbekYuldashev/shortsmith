"""Optional exercise, Step 7.2: the rules behind `--budget` and the exit codes, in plain Python.

Phase 7 has no 🧠 step; this one is for practice. Four small functions are the core of
src/shortsmith/budget.py and cli.py:
1. estimate_call: what one LLM call will cost, from the ledger's history (budget.call_cost).
2. would_exceed: the stop rule checked before every node (budget.check).
3. suggest_budget: the --budget a stopped run needs to finish (cli._suggest_budget).
4. exit_code: one outcome -> one number, the same in every command (cli.ExitCode).

Write these first, then compare with the real ones.
Check them from the repo root:  uv run pytest exercises/phase-7
Every number here is small enough to check on paper.
"""

import math  # noqa: F401  (you will want math.ceil)

WINDOW = 10  # past calls looked at


def estimate_call(past: list[float], first_call: float, window: int = WINDOW) -> float:
    """The estimated list price of the next call: the MOST expensive of the last `window` calls in
    `past` (oldest first), or `first_call` when there is no history.

    Example: past [0.009, 0.001, 0.002], window 2 -> 0.002 (0.009 is outside the window)
             past [], first_call 0.004 -> 0.004

    Hints:
    1. The last `window` items of a list: past[-window:].
    2. Why the maximum and not the average? A budget is a limit: an estimate that is usually too
       high stops a run a little early, one that is usually right overshoots half the time.
    """
    raise NotImplementedError("your turn: follow the hints above")


def would_exceed(spent: float, estimate: float, limit: float | None) -> bool:
    """Would running the next step take the run over its budget?

    - no budget (limit None): never;
    - a free step (estimate 0): never, even when the run is already over (stopping it saves nothing);
    - otherwise: when spent + estimate is MORE than the limit (exactly equal still fits).

    Example: spent 0.004, estimate 0.001, limit 0.005 -> False;  limit 0.0049 -> True
    """
    raise NotImplementedError("your turn")


def suggest_budget(spent: float, remaining: list[float]) -> float:
    """A budget that would let a stopped run finish: what it spent plus every remaining estimate,
    rounded UP to a whole cent, and at least one cent.

    Example: spent 0.0005, remaining [0.01, 0.01, 0.001] -> 0.0215 -> 0.03
             spent 0, remaining [] -> 0.01

    Hint: rounding up to cents is math.ceil(x * 100) / 100. (Careful: 0.1 + 0.2 is not exactly
    0.3 in floating point; the tests only use numbers where this doesn't matter.)
    """
    raise NotImplementedError("your turn")


EXIT_CODES = {"ok": 0, "error": 1, "usage": 2, "paused": 3, "over_budget": 4, "interrupted": 130}


def exit_code(outcome: str) -> int:
    """The exit code for an outcome, from EXIT_CODES. An unknown outcome is a bug in the caller:
    raise ValueError (with the outcome in the message) instead of guessing a number.

    Why fixed numbers? A script (or Phase 8's API) can then tell "paused, needs an answer" (3) from
    "over budget" (4) from "something broke" (1) without reading the text.
    """
    raise NotImplementedError("your turn")
