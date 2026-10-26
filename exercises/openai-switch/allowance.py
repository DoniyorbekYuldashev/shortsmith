"""Optional exercise, the OpenAI switch: what OpenAI's free daily tokens make a call cost, in plain Python.

OpenAI gives an organization that shares its API traffic a free allowance of tokens a day per model
GROUP ("mini" models share one allowance, "large" models another), refreshed at 00:00 UTC. Two small
functions are the core of src/shortsmith/costs.py:
1. allowance_day: which allowance a call belongs to (costs.allowance_day).
2. billed: what each call is billed, walking the calls in time order (costs.Billing).

Write these first, then compare with the real ones.
Check them from the repo root:  uv run pytest exercises/openai-switch
Every number here is small enough to check on paper.
"""

from datetime import date, datetime

# One call: (when, group, tokens, list price in USD). group is "mini", "large", or None (never free).
Call = tuple[datetime, str | None, int, float]


def allowance_day(ts: datetime) -> date:
    """The UTC date of `ts` (an aware datetime, in any time zone).

    Example: 2026-10-27 04:59 in Tashkent (UTC+5) -> 2026-10-26: still the 26th's allowance.

    Hint: datetime.astimezone(timezone.utc) moves a moment to UTC without changing the moment.
    """
    raise NotImplementedError("your turn: follow the hint above")


def billed(calls: list[Call], daily: dict[str, int]) -> list[float]:
    """What each call in `calls` is billed, in the same order as `calls` (which may be in ANY order).

    Within one UTC day and one group, the first `daily[group]` tokens are free. A call that crosses the
    line is billed for the share of its tokens beyond it: 800 tokens, 400 of them past the line, at a list
    price of $0.006 -> $0.003. A call with group None is billed its whole list price.

    Example: daily {"mini": 1000}; calls (in time order) of 600, 800 and 200 mini tokens at $0.004, $0.006
    and $0.002 -> [0.0, 0.003, 0.002].

    Hints:
    1. Sort the indexes by time (sorted(range(len(calls)), key=...)), so you can still answer in the
       callers' order.
    2. Keep a running total per (day, group) in a dict; free = max(0, min(tokens, daily - used_before)).
    3. Why walk the whole ledger and not only the lines being summed? Whether a call was free depends on
       every call of its group earlier that day, including other runs' calls.
    """
    raise NotImplementedError("your turn: follow the hints above")
