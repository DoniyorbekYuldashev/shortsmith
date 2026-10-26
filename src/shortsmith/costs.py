"""What the ledger says was spent (Step 7.1, `shortsmith cost`): by day, by run, by model and step.

Two numbers per line. LIST price is what ledger.py records: the paid-tier price, even for a call that
was free. BILLED is what money actually left:
- Gemini on the free tier: $0. The ledger doesn't store which tier answered a call (the tier belongs to
  the Google AI Studio project, not to a request), so this follows GEMINI_TIER: "free" (the default:
  this project's key has no billing account) or "paid".
- OpenAI text models with data sharing on (OPENAI_DATA_SHARING=1): $0 within the day's complimentary
  tokens, list price beyond them. OpenAI gives a shared daily allowance per model GROUP ("mini" models:
  2.5M tokens, "large": 250K at usage tiers 1-2), input and output counted together, refreshed at 00:00
  UTC. Billing walks the ledger in time order and keeps a running total per UTC day and group; the call
  that crosses the line is billed for the share of its tokens beyond it.
- Everything else (embeddings, images, fal.ai clips, OpenAI without data sharing): the list price.

Quotas, honestly: no API tells us how much of a free daily quota is left, so `cost` shows what OUR
logs recorded since the quota day began:
- Gemini (midnight Pacific): ledger lines per model. Only calls that got an answer are in the ledger; a
  refused request (429) is not, and Google may count it anyway. The daily limits are the ones measured in Phase 6.
- OpenAI (00:00 UTC): tokens per group from our ledger. OpenAI's own counter can differ: calls made
  outside this project with the same organization (it is shared with Yukdosh) count there and not here,
  and which models are in which group is shown only on the organization's Data controls -> Sharing page.
- YouTube: units from runs/youtube_quota.jsonl, which logs every request, refused ones included.
"""

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from shortsmith import ledger
from shortsmith.research import youtube

# Requests a day per model on the free tier, as measured (Phase 6 build notes, E11). flash-lite allows
# far more; its exact number isn't known from our side (aistudio.google.com/rate-limit shows it).
FREE_DAILY_REQUESTS = {"gemini-3.8-flash": 20, "gemini-3.6-flash": 20}
NO_RUN = "(no run id)"  # lines from before Phase 6, demos and evaluations

# OpenAI's data-sharing groups: each shares one daily allowance. From OpenAI's help page and the Sharing
# page's popup as users quoted it (2025-2026); only that page says for sure. Not listed = never free
# (embeddings, images, gpt-5.5...). gpt-5.4-mini/-nano are listed as "mini", but users reported them
# billed against the large group's cap (OpenAI community forum, March 2026): not a default here.
FREE_GROUPS: dict[str, Literal["mini", "large"]] = {
    **dict.fromkeys(["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.1-codex-mini", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini",
                     "gpt-4.1-nano", "gpt-4o-mini", "o4-mini", "o3-mini", "o1-mini", "codex-mini-latest"], "mini"),
    **dict.fromkeys(["gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5.1-codex", "gpt-5", "gpt-5-codex", "gpt-5-chat-latest",
                     "gpt-4.1", "gpt-4o", "o3", "o1"], "large"),
}
# Tokens a day per group at usage tiers 1-2 (tiers 3-5: 10M and 1M). Settings, not facts: the Sharing page shows yours.
FREE_DAILY_TOKENS = {"mini": 2_500_000, "large": 250_000}
ALLOWANCE_SETTINGS = ("OPENAI_DATA_SHARING", "OPENAI_FREE_DAILY_TOKENS_MINI", "OPENAI_FREE_DAILY_TOKENS_LARGE",
                      "OPENAI_FREE_LIMIT_ACTION")


def free_tier() -> bool:
    return os.environ.get("GEMINI_TIER", "free").strip().lower() != "paid"


class Allowance(BaseModel):
    """OpenAI's complimentary daily tokens, as the settings describe them."""

    data_sharing: bool  # OPENAI_DATA_SHARING=1: turned on for the organization (Data controls -> Sharing)
    daily_tokens: dict[str, int]  # group -> tokens a day
    action: Literal["warn", "stop"]  # before a node would pass the allowance (budget.py)
    configured: dict[str, bool]  # which of ALLOWANCE_SETTINGS the environment sets (the rest are defaults)


def allowance() -> Allowance:
    def tokens(group: str) -> int:
        value = os.environ.get(f"OPENAI_FREE_DAILY_TOKENS_{group.upper()}", "").strip().replace("_", "")
        return int(value) if value else FREE_DAILY_TOKENS[group]

    action = os.environ.get("OPENAI_FREE_LIMIT_ACTION", "warn").strip().lower()
    if action not in ("warn", "stop"):
        raise ValueError(f"OPENAI_FREE_LIMIT_ACTION={action!r}, expected warn or stop")
    return Allowance(data_sharing=os.environ.get("OPENAI_DATA_SHARING", "0").strip().lower() in ("1", "true", "on", "yes"),
                     daily_tokens={g: tokens(g) for g in FREE_DAILY_TOKENS}, action=action,  # type: ignore[arg-type]
                     configured={name: bool(os.environ.get(name, "").strip()) for name in ALLOWANCE_SETTINGS})


def free_group(model: str) -> str | None:
    """"mini", "large" or None (never free) for a model, dated snapshot names included."""
    return FREE_GROUPS.get(ledger.base_model(model))


def allowance_day(ts: datetime) -> date:
    """OpenAI's quota day: the UTC date (the allowance refreshes at 00:00 UTC)."""
    return ts.astimezone(timezone.utc).date()


def tokens_of(e: ledger.LedgerEntry) -> int:
    """Input and output together, as the allowance counts them (thinking included)."""
    return e.prompt_tokens + e.completion_tokens + e.reasoning_tokens


def _key(e: ledger.LedgerEntry) -> tuple:
    return e.ts, e.step, e.model, e.prompt_tokens, e.completion_tokens


class Billing:
    """What each ledger line was billed. Built once over the WHOLE ledger: whether an OpenAI call was free
    depends on every call of its group earlier that UTC day, not only on the lines being summed."""

    def __init__(self, lines: list[ledger.LedgerEntry], settings: Allowance | None = None):
        settings = settings or allowance()
        self.free_share: dict[tuple, float] = {}  # line -> the share of its tokens the allowance covered
        if not settings.data_sharing:
            return
        used: dict[tuple[date, str], int] = {}
        for e in sorted(lines, key=lambda e: e.ts):
            group = free_group(e.model)
            if group is None:
                continue
            day = (allowance_day(e.ts), group)
            before, tokens = used.get(day, 0), tokens_of(e)
            free = max(0, min(tokens, settings.daily_tokens[group] - before))
            used[day] = before + tokens
            self.free_share[_key(e)] = free / tokens if tokens else 1.0

    def __call__(self, e: ledger.LedgerEntry) -> float:
        if e.model.startswith("gemini") and free_tier():
            return 0.0
        return e.cost_usd * (1 - self.free_share.get(_key(e), 0.0))

    def total(self, lines: list[ledger.LedgerEntry]) -> float:
        return sum(self(e) for e in lines)


def billed_usd(entry: ledger.LedgerEntry) -> float:
    """One line on its own (no earlier calls that day): Billing over the whole ledger is the real answer."""
    return Billing([entry])(entry)


def billing_note() -> str:
    """One line on what "billed" means with the current settings (the CLI and the web page show it)."""
    from shortsmith import llm

    a = allowance()
    name = llm.provider_name()
    if name == "gemini":
        return ("Gemini free tier: LLM calls billed $0 (costs shown are list price)" if free_tier()
                else "GEMINI_TIER=paid: Gemini calls are billed at list price")
    if a.data_sharing:
        return (f"OpenAI data sharing: text calls billed $0 within today's free tokens ({a.daily_tokens['mini']:,} mini, "
                f"{a.daily_tokens['large']:,} large; 00:00 UTC), list price beyond; embeddings and images at list price")
    return "OpenAI without data sharing (OPENAI_DATA_SHARING is off): every call is billed at list price"


class CostRow(BaseModel):
    key: str  # the day, the run id, or "model | step"
    model: str | None = None
    step: str | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0  # visible output + thinking
    list_usd: float = 0.0
    billed_usd: float = 0.0

    def add(self, e: ledger.LedgerEntry, billed: float) -> None:
        self.calls += 1
        self.prompt_tokens += e.prompt_tokens
        self.completion_tokens += e.completion_tokens + e.reasoning_tokens
        self.list_usd += e.cost_usd
        self.billed_usd += billed


class ModelToday(BaseModel):
    model: str
    requests: int  # ledger lines since midnight Pacific (answered calls only)
    daily_limit: int | None  # free-tier requests a day, as measured; None = not known


class GroupToday(BaseModel):
    """One OpenAI data-sharing group since 00:00 UTC, by our ledger."""

    group: str  # "mini" or "large"
    models: list[str]  # the group's models used today
    tokens: int  # input + output, every call of the group
    daily_tokens: int  # the allowance (a setting)
    left: int  # by our count; OpenAI's own counter may say less (see the notes)
    billed_usd: float  # what today's calls beyond the allowance cost


class AllowanceToday(BaseModel):
    data_sharing: bool
    day_start: datetime  # 00:00 UTC, when OpenAI refreshes the complimentary tokens
    action: Literal["warn", "stop"]
    groups: list[GroupToday]
    notes: list[str]


class Quotas(BaseModel):
    day_start: datetime  # midnight Pacific time, when Google resets both quotas (in UTC)
    gemini: list[ModelToday]
    openai: AllowanceToday
    youtube_units: int  # today, from our quota log
    youtube_cap: int  # our own cap (youtube.DAILY_LIMIT_UNITS)
    youtube_limit: int  # Google's free quota
    notes: list[str]


class CostReport(BaseModel):
    since: date | None  # None = the whole ledger
    run: str | None  # only this run's lines
    free_tier: bool  # Gemini's free tier (GEMINI_TIER)
    billing: str  # what "billed" means with these settings, in one line
    total: CostRow
    by_day: list[CostRow]  # newest first, in this computer's time zone
    by_run: list[CostRow]  # newest first
    by_model_step: list[CostRow]  # most expensive first
    quotas: Quotas


def quota_day_start(now: datetime) -> datetime:
    pacific = now.astimezone(youtube.PACIFIC)
    return pacific.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def used_today(lines: list[ledger.LedgerEntry], now: datetime) -> dict[str, int]:
    """Tokens per data-sharing group since 00:00 UTC (OpenAI's day), from our ledger."""
    today = allowance_day(now)
    used = dict.fromkeys(FREE_DAILY_TOKENS, 0)
    for e in lines:
        if (group := free_group(e.model)) is not None and allowance_day(e.ts) == today:
            used[group] += tokens_of(e)
    return used


def allowance_today(lines: list[ledger.LedgerEntry], now: datetime, billing: Billing | None = None,
                    settings: Allowance | None = None) -> AllowanceToday:
    settings = settings or allowance()
    billing = billing or Billing(lines, settings)
    day = allowance_day(now)
    used = used_today(lines, now)
    groups = []
    for group, limit in settings.daily_tokens.items():
        mine = [e for e in lines if free_group(e.model) == group and allowance_day(e.ts) == day]
        groups.append(GroupToday(group=group, models=sorted({ledger.base_model(e.model) for e in mine}),
                                 tokens=used[group], daily_tokens=limit, left=max(0, limit - used[group]),
                                 billed_usd=billing.total(mine)))
    notes = [
        f"OpenAI's own counter can't be asked: these are the tokens OUR ledger recorded since 00:00 UTC "
        f"({datetime.combine(day, datetime.min.time(), timezone.utc):%Y-%m-%d %H:%M} UTC). Calls made outside this "
        "project with the same organization (it is shared with Yukdosh) count there, not here.",
        "Which models are free and how many tokens a day: only the organization's Data controls -> Sharing page "
        "says for sure (OPENAI_FREE_DAILY_TOKENS_MINI / _LARGE are settings, not facts).",
    ]
    if not settings.data_sharing:
        notes.insert(0, "OPENAI_DATA_SHARING is off: every OpenAI call is billed at list price.")
    return AllowanceToday(data_sharing=settings.data_sharing, action=settings.action,
                          day_start=datetime.combine(day, datetime.min.time(), timezone.utc), groups=groups, notes=notes)


def quotas(lines: list[ledger.LedgerEntry], now: datetime, critic_model: str | None = None,
           quota_log: Path = youtube.QUOTA_LOG, billing: Billing | None = None) -> Quotas:
    start = quota_day_start(now)
    today: dict[str, int] = {model: 0 for model in FREE_DAILY_REQUESTS}
    if critic_model and critic_model.startswith("gemini"):
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
                  openai=allowance_today(lines, now, billing),
                  youtube_units=youtube.units_today(quota_log, now), youtube_cap=youtube.DAILY_LIMIT_UNITS,
                  youtube_limit=youtube.GOOGLE_DAILY_UNITS, notes=notes)


def report(lines: list[ledger.LedgerEntry], days: int | None = None, run: str | None = None,
           now: datetime | None = None, critic_model: str | None = None,
           quota_log: Path = youtube.QUOTA_LOG) -> CostReport:
    """The ledger grouped three ways. `days`: only the last N days (today included); `run`: one run.
    `lines` must be the whole ledger even then: billing an OpenAI call needs its day's earlier calls."""
    now = now or datetime.now(timezone.utc)
    billing = Billing(lines)
    since = (now.astimezone().date() - timedelta(days=days - 1)) if days else None
    chosen = [e for e in lines if (since is None or e.ts.astimezone().date() >= since)
              and (run is None or e.run_id == run)]
    total = CostRow(key="total")
    by_day: dict[str, CostRow] = {}
    by_run: dict[str, CostRow] = {}
    by_model_step: dict[str, CostRow] = {}
    for e in chosen:
        billed = billing(e)
        total.add(e, billed)
        day = e.ts.astimezone().date().isoformat()
        by_day.setdefault(day, CostRow(key=day)).add(e, billed)
        key = e.run_id or NO_RUN
        by_run.setdefault(key, CostRow(key=key)).add(e, billed)
        key = f"{e.model} | {e.step}"
        by_model_step.setdefault(key, CostRow(key=key, model=e.model, step=e.step)).add(e, billed)
    return CostReport(
        since=since, run=run, free_tier=free_tier(), billing=billing_note(), total=total,
        by_day=sorted(by_day.values(), key=lambda r: r.key, reverse=True),
        by_run=sorted(by_run.values(), key=lambda r: (r.key != NO_RUN, r.key), reverse=True),
        by_model_step=sorted(by_model_step.values(), key=lambda r: r.list_usd, reverse=True),
        quotas=quotas(lines, now, critic_model, quota_log, billing),
    )
