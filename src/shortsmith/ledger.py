"""Cost ledger: one JSON line per paid call (LLM calls now, images and video later).

Append-only JSON Lines: a crash can damage at most the last line, and pandas reads it
with `pd.read_json(path, lines=True)`. MLflow (Phase 6) only watches runs; this file is
what the `--budget` cap will check to stop one.
"""

import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_PATH = Path("runs/ledger.jsonl")  # relative to where commands run (the repo root)

# The graph run (Phase 6) whose node is making calls right now. A context variable instead of a
# new argument on every llm/media function: each graph node sets it, and every line recorded
# while it runs is tagged, so "what did run X pay for?" is one filter (the resume proof uses it).
CURRENT_RUN: ContextVar[str | None] = ContextVar("shortsmith_run", default=None)

# USD per generated item, for media priced per output instead of per token. FLUX schnell on
# fal: ~$0.003 per image up to 1 megapixel (plan estimate; verify on fal's model page before
# the first real run).
MEDIA_PRICES: dict[str, float] = {
    "fal-ai/flux/schnell": 0.003,
    # Nano Banana (Google's Gemini image model) on fal: $0.0398 per image (CLAUDE.md, 2026-10-25).
    "fal-ai/nano-banana": 0.0398,
    # One 5-second image-to-video hero clip (Phase 6). An ESTIMATE, never charged yet: PLAN §3
    # puts real models at ~$0.25-0.60 per 5-s clip. Check the live price before the first one.
    "fal-ai/kling-video/v2.1/standard/image-to-video": 0.30,
}

# OpenAI's image models are priced per TOKEN (USD per 1M): the prompt as text input, the picture as
# image output. developers.openai.com/api/docs/pricing, checked 2026-10-26. A response carries its
# own usage, so an image is priced from what it really used (media/images.py).
IMAGE_TOKEN_PRICES: dict[str, dict[str, float]] = {
    "gpt-image-2": {"text_input": 5.00, "image_input": 8.00, "output": 30.00},
    "gpt-image-1-mini": {"text_input": 2.00, "image_input": 2.50, "output": 8.00},
}
# USD per 1024x1536 image by quality, for estimates before anything is made (and for a response
# without usage). OpenAI's reference figures as quoted by third parties (its own calculator isn't
# a page we can read): gpt-image-2 low ~$0.005, medium ~$0.041, high ~$0.165. gpt-image-1-mini:
# OpenAI's published table. Other sizes are scaled by pixel count (images.price_per_image).
IMAGE_PRICES: dict[str, dict[str, float]] = {
    "gpt-image-2": {"low": 0.005, "medium": 0.041, "high": 0.165},
    "gpt-image-1-mini": {"low": 0.006, "medium": 0.015, "high": 0.052},
}

# USD per 1,000,000 tokens. These are LIST prices, recorded even when data sharing makes a
# call free, so budgets stay honest. OpenAI's: developers.openai.com/api/docs/pricing, checked
# 2026-10-26 (standard tier, prompts under 272K tokens).
PRICES: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    # Gemini paid-tier list prices, from ai.google.dev/gemini-api/docs/pricing (checked
    # 2026-10-21). Our key is on the free tier, so these calls really cost $0. Output
    # includes thinking tokens. 3.8 Flash doubles on 2027-01-01.
    "gemini-3.5-flash-lite": {"input": 0.30, "cached_input": 0.03, "output": 2.50},
    "gemini-3.8-flash": {"input": 0.75, "cached_input": 0.075, "output": 3.75},
    # The critic's fallback when 3.8 Flash's free daily quota (20 requests per model) is used up
    # (Phase 6 build notes, E11). Same list price, same page, checked 2026-10-24; doubles on 2027-01-01.
    "gemini-3.6-flash": {"input": 0.75, "cached_input": 0.075, "output": 3.75},
    # Embeddings: input only (a vector is not generated text). gemini-embedding-2: $0.20 per 1M
    # text tokens on the paid tier, free on ours (same pricing page, checked 2026-10-23).
    # text-embedding-3-small/-large: OpenAI's list prices (same page as the chat models).
    "gemini-embedding-2": {"input": 0.20, "cached_input": 0.20, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "cached_input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "cached_input": 0.13, "output": 0.0},
}

_SNAPSHOT_DATE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def base_model(model: str) -> str:
    """The model a dated snapshot belongs to: "gpt-5-mini-2025-08-07" -> "gpt-5-mini".

    Only a trailing date is cut, never a prefix match: "gpt-5" priced every name starting with it would
    price gpt-5.5 (4x dearer) and gpt-5-pro (12x) as gpt-5 without a word. An unknown name must fail."""
    return _SNAPSHOT_DATE.sub("", model)


class LedgerEntry(BaseModel):
    """One paid call."""

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    step: str  # who made the call: "writer", "critic", "demo"...
    model: str  # exact version the API reported, e.g. "gpt-4.1-mini-2025-04-14"
    prompt_tokens: int
    cached_tokens: int = 0
    completion_tokens: int
    reasoning_tokens: int = 0  # thinking billed as output that the provider reported outside completion_tokens
    cost_usd: float
    # True when the provider reported no usage and the tokens were counted locally with
    # tiktoken (Gemini's OpenAI-compatible embeddings endpoint sends usage: null).
    tokens_estimated: bool = False
    run_id: str | None = Field(default_factory=CURRENT_RUN.get)  # set inside a graph run; None elsewhere


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """The list price of one call, in USD.

    The API reports dated versions ("gpt-4.1-mini-2025-04-14"), so the price is looked up by
    base_model(). An unknown model raises instead of costing $0: a silent $0 would let a run
    blow straight through the --budget cap.
    """
    price = PRICES.get(base_model(model))
    if price is None:
        raise ValueError(f"No price for model {model!r}: add it to PRICES in ledger.py")

    uncached = prompt_tokens - cached_tokens  # cached tokens are part of the prompt
    return (
        uncached * price["input"]
        + cached_tokens * price["cached_input"]
        + completion_tokens * price["output"]
    ) / 1_000_000


def record(entry: LedgerEntry, path: Path) -> None:
    """Append `entry` as one JSON line; earlier lines are never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def entries(path: Path, run_id: str | None = None) -> list[LedgerEntry]:
    """Every line, or only one graph run's. A missing file means nothing was spent yet.

    Parsed with CURRENT_RUN cleared: run_id's default_factory also runs for a line that has no
    run_id (every line from before Phase 6), and inside a node it would claim all of them for the
    current run. That made a new run look $0.066 spent at its first budget check (Phase 7, E-ledger)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    token = CURRENT_RUN.set(None)
    try:
        found = [LedgerEntry.model_validate_json(line) for line in lines if line.strip()]
    finally:
        CURRENT_RUN.reset(token)
    return [e for e in found if run_id is None or e.run_id == run_id]


def total_cost(path: Path) -> float:
    """Sum of cost_usd over every line. A missing file means nothing was spent yet."""
    if not path.exists():
        return 0.0
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(LedgerEntry.model_validate_json(line).cost_usd for line in lines if line.strip())
