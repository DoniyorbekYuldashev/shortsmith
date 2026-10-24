"""What the ledger says was spent (Step 7.1, `shortsmith cost`): by day, by run, by model and step.

Two numbers per line. LIST price is what ledger.py records: the paid-tier price, even for a call that
was free. BILLED is what money actually left: Gemini calls on the free tier are $0; everything else
(OpenAI, fal.ai images and clips) is billed at its list price. The ledger doesn't store which tier
answered a call (the tier belongs to the Google AI Studio project, not to a request), so `billed`
follows GEMINI_TIER: "free" (the default: this project's key has no billing account) or "paid".

Quotas, honestly: no API tells us how much of the free tier's daily quota is left, so `cost` shows
what OUR logs recorded since the quota day began (midnight Pacific time):
- Gemini: ledger lines per model. Only calls that got an answer are in the ledger; a refused request
  (429) is not, and Google may count it anyway. The daily limits are the ones measured in Phase 6.
- YouTube: units from runs/youtube_quota.jsonl, which logs every request, refused ones included.
"""

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

from shortsmith import ledger
from shortsmith.research import youtube

# Requests a day per model on the free tier, as measured (Phase 6 build notes, E11). flash-lite allows
# far more; its exact number isn't known from our side (aistudio.google.com/rate-limit shows it).
FREE_DAILY_REQUESTS = {"gemini-3.8-flash": 20, "gemini-3.6-flash": 20}
NO_RUN = "(no run id)"  # lines from before Phase 6, demos and evaluations


def free_tier() -> bool:
    return os.environ.get("GEMINI_TIER", "free").strip().lower() != "paid"


def billed_usd(entry: ledger.LedgerEntry) -> float:
    """What a line really cost: $0 for Gemini on the free tier, the list price for everything else."""
    return 0.0 if entry.model.startswith("gemini") and free_tier() else entry.cost_usd


class CostRow(BaseModel):
    key: str  # the day, the run id, or "model | step"
    model: str | None = None
    step: str | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0  # visible output + thinking
    list_usd: float = 0.0
    billed_usd: float = 0.0

    def add(self, e: ledger.LedgerEntry) -> None:
        self.calls += 1
        self.prompt_tokens += e.prompt_tokens
        self.completion_tokens += e.completion_tokens + e.reasoning_tokens
        self.list_usd += e.cost_usd
        self.billed_usd += billed_usd(e)


class ModelToday(BaseModel):
    model: str
    requests: int  # ledger lines since midnight Pacific (answered calls only)
    daily_limit: int | None  # free-tier requests a day, as measured; None = not known


class Quotas(BaseModel):
    day_start: datetime  # midnight Pacific time, when Google resets both quotas (in UTC)
    gemini: list[ModelToday]
    youtube_units: int  # today, from our quota log
    youtube_cap: int  # our own cap (youtube.DAILY_LIMIT_UNITS)
    youtube_limit: int  # Google's free quota
    notes: list[str]


class CostReport(BaseModel):
    since: date | None  # None = the whole ledger
    run: str | None  # only this run's lines
    free_tier: bool
    total: CostRow
    by_day: list[CostRow]  # newest first, in this computer's time zone
    by_run: list[CostRow]  # newest first
    by_model_step: list[CostRow]  # most expensive first
    quotas: Quotas


def quota_day_start(now: datetime) -> datetime:
    pacific = now.astimezone(youtube.PACIFIC)
    return pacific.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def quotas(lines: list[ledger.LedgerEntry], now: datetime, critic_model: str | None = None,
           quota_log: Path = youtube.QUOTA_LOG) -> Quotas:
    start = quota_day_start(now)
    today: dict[str, int] = {model: 0 for model in FREE_DAILY_REQUESTS}
    if critic_model:
        today.setdefault(critic_model, 0)
    for e in lines:
        if e.model.startswith("gemini") and e.ts >= start:
            today[e.model] = today.get(e.model, 0) + 1
    notes = [
        "Google can't be asked how much free quota is left: these are the calls OUR logs recorded since "
        f"midnight Pacific ({start:%Y-%m-%d %H:%M} UTC).",
        "Refused requests (429) are not in the ledger but may still count against the day.",
        "flash-lite's daily limit is far higher than Flash's 20 and not measured; see aistudio.google.com/rate-limit.",
    ]
    return Quotas(day_start=start, gemini=[ModelToday(model=m, requests=n, daily_limit=FREE_DAILY_REQUESTS.get(m))
                                           for m, n in sorted(today.items())],
                  youtube_units=youtube.units_today(quota_log, now), youtube_cap=youtube.DAILY_LIMIT_UNITS,
                  youtube_limit=youtube.GOOGLE_DAILY_UNITS, notes=notes)


def report(lines: list[ledger.LedgerEntry], days: int | None = None, run: str | None = None,
           now: datetime | None = None, critic_model: str | None = None,
           quota_log: Path = youtube.QUOTA_LOG) -> CostReport:
    """The ledger grouped three ways. `days`: only the last N days (today included); `run`: one run."""
    now = now or datetime.now(timezone.utc)
    since = (now.astimezone().date() - timedelta(days=days - 1)) if days else None
    chosen = [e for e in lines if (since is None or e.ts.astimezone().date() >= since)
              and (run is None or e.run_id == run)]
    total = CostRow(key="total")
    by_day: dict[str, CostRow] = {}
    by_run: dict[str, CostRow] = {}
    by_model_step: dict[str, CostRow] = {}
    for e in chosen:
        total.add(e)
        day = e.ts.astimezone().date().isoformat()
        by_day.setdefault(day, CostRow(key=day)).add(e)
        key = e.run_id or NO_RUN
        by_run.setdefault(key, CostRow(key=key)).add(e)
        key = f"{e.model} | {e.step}"
        by_model_step.setdefault(key, CostRow(key=key, model=e.model, step=e.step)).add(e)
    return CostReport(
        since=since, run=run, free_tier=free_tier(), total=total,
        by_day=sorted(by_day.values(), key=lambda r: r.key, reverse=True),
        by_run=sorted(by_run.values(), key=lambda r: (r.key != NO_RUN, r.key), reverse=True),
        by_model_step=sorted(by_model_step.values(), key=lambda r: r.list_usd, reverse=True),
        quotas=quotas(lines, now, critic_model, quota_log),
    )
