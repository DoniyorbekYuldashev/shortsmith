"""Thin wrapper around the Chat Completions format.

Every agent calls the LLM through `chat()`, so tokens, cost and latency are measured in
one place, and every call lands in the cost ledger. OpenAI and Gemini both speak this
format, so switching provider only changes the address, the key and the model name.
"""

import argparse
import os
import time
from functools import cache
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from shortsmith import ledger
from shortsmith.models import NARRATION_WORDS, SCENES_PLANNED, Script

DEMO_SYSTEM_PROMPT = "You are a short-video scriptwriter. Be concise."  # also used by tokens.py's prediction
DEFAULT_LEDGER = ledger.DEFAULT_PATH


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


# One chat message in the API's own wire format. A plain dict on purpose: an assistant message
# with tool calls nests objects, and Gemini adds fields of its own that must go back unchanged.
Message = dict[str, Any]


class ToolCall(BaseModel):
    """One function call the model asks for. The model only asks: our code decides and runs it."""

    id: str  # the tool message that carries the result must quote this id
    name: str
    arguments: str  # JSON text written by the model: checked by the tool's args model, never trusted


class ChatResult(BaseModel):
    """One LLM call: what came back and what it consumed."""

    text: str
    tool_calls: list[ToolCall] = []  # empty = a plain answer (for an agent: "I'm done")
    message: Message = Field(default_factory=dict, repr=False)  # the assistant message exactly as returned
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


def _complete(
    messages: list[Message],
    model: str | None,
    step: str,
    ledger_path: Path,
    temperature: float | None = None,
    **request: object,
) -> ChatResult:
    """The one place that sends a request: every call, plain or JSON, is priced and recorded here."""
    model = model or provider().default_model
    ledger.cost_usd(model, 0, 0)  # fail before paying if this model has no price
    if temperature is not None:  # not `if temperature:`, because 0.0 is a real choice
        request["temperature"] = temperature  # left out = the provider's default (1.0 for Gemini 3)

    start = time.perf_counter()
    response = _client().chat.completions.create(model=model, messages=messages, **request)
    latency = time.perf_counter() - start

    usage = response.usage
    cached = (usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0) or 0
    answered_by = response.model.removeprefix("models/")  # Gemini names models "models/gemini-..."
    message = response.choices[0].message
    result = ChatResult(
        text=message.content or "",  # None when the model refuses or only calls tools
        tool_calls=[
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in message.tool_calls or []
            if call.type == "function"
        ],
        # model_dump keeps fields the SDK doesn't know: Gemini 3 puts a "thought signature" on
        # tool calls, and the next request is checked for it. Rebuilding the dict would drop it.
        message=message.model_dump(exclude_none=True),
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


def chat(
    messages: list[Message],
    model: str | None = None,
    step: str = "adhoc",
    ledger_path: Path = DEFAULT_LEDGER,
    temperature: float | None = None,
) -> ChatResult:
    """Send one request and record it. The model is stateless: `messages` must hold the whole conversation."""
    return _complete(messages, model, step, ledger_path, temperature)


def chat_tools(
    messages: list[Message],
    tools: list[Message],
    model: str | None = None,
    step: str = "agent",
    ledger_path: Path = DEFAULT_LEDGER,
    temperature: float | None = None,
) -> ChatResult:
    """One agent turn: the model sees the tool schemas and may answer with tool calls instead of text.

    Nothing runs here. `result.tool_calls` holds names and JSON arguments; executing them and
    sending the results back is the caller's job (agents/loop.py). The schemas are sent with
    every turn and count as prompt tokens each time.
    """
    return _complete(messages, model, step, ledger_path, temperature, tools=tools)


def chat_json[T: BaseModel](
    messages: list[Message],
    schema: type[T],
    model: str | None = None,
    step: str = "adhoc",
    ledger_path: Path = DEFAULT_LEDGER,
    temperature: float | None = None,
) -> tuple[T, ChatResult]:
    """Ask for JSON that matches `schema`, record the call, then validate it.

    The API enforces the schema's shape; the schema's own validators enforce the rules.
    Validation runs after the ledger write because a call is paid even if its answer fails.
    (The SDK's `.parse()` does the same in one call, but validates before we could record.)
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
    }
    result = _complete(messages, model, step, ledger_path, temperature, response_format=response_format)
    return schema.model_validate_json(result.text), result


LANGUAGES = {"en": "English", "uz": "Uzbek (Latin script)", "ru": "Russian"}

# Word targets come from models.NARRATION_WORDS, so the prompt and the validator can't disagree.
# Per scene as well as in total: an LLM counts 10 words far better than 60 (Step 2.4: asked
# only for a total, Gemini wrote 34 Uzbek words instead of 55-60).
_WORD_TARGETS = "\n".join(
    f"  - {LANGUAGES[code]}: {round(low / SCENES_PLANNED)}-{round(high / SCENES_PLANNED)} words per scene, "
    f"{low}-{high} words in total"
    for code, (low, high) in NARRATION_WORDS.items()
)

WRITER_SYSTEM_PROMPT = f"""You write scripts for vertical short videos (YouTube Shorts, Reels, TikTok).

Rules:
- Exactly {SCENES_PLANNED} scenes, each one or two short sentences.
- Length: about 30 seconds of speech. Count WORDS, not seconds. Narration length by language:
{_WORD_TARGETS}
- hook: the first spoken line, AT MOST 8 WORDS. It must make people stop scrolling.
  Scene 1's narration starts with the hook.
- cta: one short call to action. The last scene's narration ends with the cta, so it is spoken.
- narration: spoken words only (no emojis, hashtags or stage directions).
- duration_s: your estimate of how long the scene's narration takes to say.
- visual_prompt: one vertical 9:16 image, concrete and specific to the place and topic.
  End every visual_prompt with the same style words, so all scenes look consistent.
- motion: zoom_in, zoom_out, pan_left or pan_right. Never ai_clip.
- Write title, hook, narration and cta in the requested language; visual_prompt always in English.
- Number scenes with idx 1, 2, 3, ... in order."""


def write_script(topic: str, language: str = "en", model: str | None = None) -> tuple[Script, ChatResult]:
    """One topic in, one validated Script out (this becomes the Writer agent in Phase 6)."""
    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Topic: {topic}\nLanguage: {LANGUAGES[language]} (language code '{language}')"},
    ]
    return chat_json(messages, Script, model=model, step="writer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write one short-video script (the Phase 1 check).")
    parser.add_argument("topic")
    parser.add_argument("--lang", choices=sorted(LANGUAGES), default="en")
    args = parser.parse_args()
    try:
        script, result = write_script(args.topic, args.lang)
    except ValidationError as e:
        print(f"The model's script broke {e.error_count()} rule(s) (the call is still in the ledger):")
        for error in e.errors():
            print(" -", error["msg"])
        raise SystemExit(1)
    print(script.model_dump_json(indent=2))
    print(
        f"\n{result.model} | {result.prompt_tokens} in + {result.completion_tokens} out tokens | "
        f"${result.cost_usd:.6f} list price | {result.latency_s} s | ledger total "
        f"${ledger.total_cost(DEFAULT_LEDGER):.6f}"
    )
