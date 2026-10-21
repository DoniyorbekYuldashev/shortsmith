"""Thin wrapper around the Chat Completions format.

Every agent calls the LLM through `chat()`, so tokens, cost and latency are measured in
one place, and every call lands in the cost ledger. OpenAI and Gemini both speak this
format, so switching provider only changes the address, the key and the model name.
"""

import os
import sys
import time
from functools import cache
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from shortsmith import ledger

DEMO_SYSTEM_PROMPT = "You are a short-video scriptwriter. Be concise."  # also used by tokens.py's prediction
DEFAULT_LEDGER = Path("runs/ledger.jsonl")  # relative to where commands run (the repo root)


class Provider(BaseModel):
    base_url: str | None  # None = the SDK's default address (api.openai.com)
    key_env: str  # name of the environment variable that holds the API key
    default_model: str


PROVIDERS = {
    "openai": Provider(base_url=None, key_env="OPENAI_API_KEY", default_model="gpt-4.1-mini"),
    "gemini": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        default_model="gemini-3.5-flash-lite",
    ),
}


def provider() -> Provider:
    """Chosen by LLM_PROVIDER in .env; without it, OpenAI."""
    name = os.environ.get("LLM_PROVIDER", "openai")
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER={name!r}, expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[name]


class ChatResult(BaseModel):
    """One LLM call: what came back and what it consumed."""

    text: str
    model: str  # exact version that answered, e.g. "gpt-4.1-mini-2025-04-14"
    prompt_tokens: int  # input: everything we sent, system message included
    completion_tokens: int  # output: what the model wrote (the pricier rate)
    cached_tokens: int  # part of the input served from OpenAI's prompt cache (cheaper)
    cost_usd: float  # list price, from ledger.PRICES
    latency_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@cache
def _client() -> OpenAI:
    """Built on first use, so importing this module never needs an API key (tests stay offline)."""
    p = provider()
    key = os.environ.get(p.key_env)
    if not key:
        raise ValueError(f"{p.key_env} is empty: add it to .env")
    return OpenAI(api_key=key, base_url=p.base_url)


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    step: str = "adhoc",
    ledger_path: Path = DEFAULT_LEDGER,
) -> ChatResult:
    """Send one request and record it in the ledger.

    The model is stateless: `messages` must hold the whole conversation.
    """
    model = model or provider().default_model
    ledger.cost_usd(model, 0, 0)  # fail before paying if this model has no price

    start = time.perf_counter()
    response = _client().chat.completions.create(model=model, messages=messages)
    latency = time.perf_counter() - start

    usage = response.usage
    cached = (usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0) or 0
    answered_by = response.model.removeprefix("models/")  # Gemini names models "models/gemini-..."
    result = ChatResult(
        text=response.choices[0].message.content or "",  # None when the model refuses or calls a tool
        model=answered_by,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=cached,
        cost_usd=ledger.cost_usd(answered_by, usage.prompt_tokens, usage.completion_tokens, cached),
        latency_s=round(latency, 2),
    )
    ledger.record(
        ledger.LedgerEntry(
            step=step,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            cached_tokens=result.cached_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
        ),
        ledger_path,
    )
    return result


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    result = chat(
        [
            {"role": "system", "content": DEMO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        step="demo",
    )
    print(result.text)
    print()
    print(f"model              {result.model}")
    print(f"prompt tokens      {result.prompt_tokens}")
    print(f"completion tokens  {result.completion_tokens}")
    print(f"total tokens       {result.total_tokens}")
    print(f"cached tokens      {result.cached_tokens}")
    print(f"cost (list price)  ${result.cost_usd:.6f}")
    print(f"latency            {result.latency_s} s")
    print(f"ledger total       ${ledger.total_cost(DEFAULT_LEDGER):.6f}  ({DEFAULT_LEDGER})")
