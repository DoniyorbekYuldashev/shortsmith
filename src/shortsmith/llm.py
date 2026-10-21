"""Thin wrapper around OpenAI Chat Completions.

Every agent will call the LLM through `chat()`, so tokens and latency are measured
in one place. The cost ledger (Step 1.3) will hook in here.
"""

import sys
import time
from functools import cache

from openai import OpenAI
from pydantic import BaseModel

DEFAULT_MODEL = "gpt-4.1-mini"
DEMO_SYSTEM_PROMPT = "You are a short-video scriptwriter. Be concise."  # also used by tokens.py's prediction


class ChatResult(BaseModel):
    """One LLM call: what came back and what it consumed."""

    text: str
    model: str  # exact version that answered, e.g. "gpt-4.1-mini-2025-04-14"
    prompt_tokens: int  # input: everything we sent, system message included
    completion_tokens: int  # output: what the model wrote (the pricier rate)
    cached_tokens: int  # part of the input served from OpenAI's prompt cache (cheaper)
    latency_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@cache
def _client() -> OpenAI:
    """Built on first use, so importing this module never needs an API key (tests stay offline)."""
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def chat(messages: list[dict[str, str]], model: str = DEFAULT_MODEL) -> ChatResult:
    """Send one request. The model is stateless: `messages` must hold the whole conversation."""
    start = time.perf_counter()
    response = _client().chat.completions.create(model=model, messages=messages)
    latency = time.perf_counter() - start

    usage = response.usage
    cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
    return ChatResult(
        text=response.choices[0].message.content or "",  # None when the model refuses or calls a tool
        model=response.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=cached or 0,
        latency_s=round(latency, 2),
    )


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    result = chat(
        [
            {"role": "system", "content": DEMO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    print(result.text)
    print()
    print(f"model              {result.model}")
    print(f"prompt tokens      {result.prompt_tokens}")
    print(f"completion tokens  {result.completion_tokens}")
    print(f"total tokens       {result.total_tokens}")
    print(f"cached tokens      {result.cached_tokens}")
    print(f"latency            {result.latency_s} s")
