"""Thin wrapper around the Chat Completions format.

Every agent calls the LLM through `chat()`, so tokens, cost and latency are measured in
one place, and every call lands in the cost ledger. OpenAI and Gemini both speak this
format, so switching provider only changes the address, the key and the model name.
`embed()` does the same for embeddings (text -> vectors, for retrieval in research/rag.py).

Which model a call uses: the caller's, else its role's (model_for: SHORTSMITH_<ROLE>_MODEL, else
LLM_PROVIDER's defaults). Which address and key: the model's own provider (provider_of), so roles
can live on different providers, and the RAG embeddings stay on Gemini whatever LLM_PROVIDER says.
"""

import argparse
import logging
import os
import re
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import tiktoken
from openai import InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError

from shortsmith import ledger
from shortsmith.models import NARRATION_WORDS, SCENES_PLANNED, ResearchNotes, Script

DEMO_SYSTEM_PROMPT = "You are a short-video scriptwriter. Be concise."  # also used by tokens.py's prediction
DEFAULT_LEDGER = ledger.DEFAULT_PATH


class Provider(BaseModel):
    base_url: str | None  # None = the SDK's default address (api.openai.com)
    key_env: str  # name of the environment variable that holds the API key
    default_model: str  # researcher, writer, director, revise
    critic_model: str  # judges the frames: sees images, and is not the model that wrote the script
    embedding_model: str  # turns text into vectors for retrieval (research/rag.py)


PROVIDERS = {
    # gpt-5-mini: in the data-sharing "mini" group of free daily tokens (2.5M a day at usage tiers 1-2),
    # strict JSON and vision, no announced shutdown (gpt-4.1-mini goes on 2027-01-14). gpt-5.1 judges:
    # stronger than the writer, a different model, in the "large" group (250K a day at tiers 1-2).
    "openai": Provider(base_url=None, key_env="OPENAI_API_KEY", default_model="gpt-5-mini", critic_model="gpt-5.1",
                       embedding_model="text-embedding-3-small"),
    "gemini": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        default_model="gemini-3.5-flash-lite",
        critic_model="gemini-3.8-flash",  # each free Flash model allows 20 requests a day: gemini-3.6-flash is the fallback
        embedding_model="gemini-embedding-2",  # multilingual: an Uzbek query finds an English text
    ),
}


def provider_name() -> str:
    """LLM_PROVIDER in .env; without it, OpenAI."""
    name = os.environ.get("LLM_PROVIDER", "openai")
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER={name!r}, expected one of {sorted(PROVIDERS)}")
    return name


def provider() -> Provider:
    """The chat provider: its models are every role's default."""
    return PROVIDERS[provider_name()]


def provider_of(model: str) -> str:
    """Which provider serves a model: its name says so. A model always goes to its own provider's address
    and key, so SHORTSMITH_CRITIC_MODEL=gemini-3.8-flash with LLM_PROVIDER=openai still reaches Google."""
    return "gemini" if model.removeprefix("models/").startswith("gemini") else "openai"


# Ledger step -> role. Each role's model: SHORTSMITH_<ROLE>_MODEL if set (SHORTSMITH_CRITIC_MODEL is the
# Phase 6 one), else the chat provider's default (the critic: its critic_model).
ROLES = {"researcher": "researcher", "writer": "writer", "writer-pick": "writer", "director": "director",
         "critic": "critic", "revise": "revise"}


def role_model(role: str) -> str:
    if chosen := os.environ.get(f"SHORTSMITH_{role.upper()}_MODEL", "").strip():
        return chosen
    p = provider()
    return p.critic_model if role == "critic" else p.default_model


def model_for(step: str) -> str:
    """The model a call made by `step` uses when the caller names none."""
    return role_model(ROLES[step]) if step in ROLES else provider().default_model


def embedding_model() -> str:
    """The model behind the RAG index: always Gemini's (owner's choice, 2026-10-26), whatever LLM_PROVIDER says.
    The index belongs to one embedding model (kb_gemini_embedding_2_768), so switching the text provider to
    OpenAI must not move it; provider_of() sends it to Google's address with Google's key."""
    return PROVIDERS["gemini"].embedding_model


# OpenAI's reasoning models think before answering; how much is a request setting. "low" for every role:
# fast, and enough for scripts and a critique. Gemini is left at its default (nothing sent).
REASONING_EFFORT_DEFAULT = "low"
_REASONING_FAMILIES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def reasoning_effort(model: str) -> str | None:
    """What to send as reasoning_effort for `model`: OPENAI_REASONING_EFFORT ("" = the API's default),
    only to OpenAI's reasoning models (a chat-latest alias doesn't take it)."""
    if provider_of(model) != "openai" or not model.startswith(_REASONING_FAMILIES) or "chat" in model:
        return None
    return os.environ.get("OPENAI_REASONING_EFFORT", REASONING_EFFORT_DEFAULT).strip() or None


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
    # Thinking the model did before answering, billed as output. OpenAI counts it inside
    # completion_tokens; Gemini's compatible endpoint leaves it out and only adds it to total_tokens.
    reasoning_tokens: int = 0
    cost_usd: float  # list price, from ledger.PRICES
    latency_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


@cache
def _client(name: str | None = None) -> OpenAI:
    """One client per provider (`name`; None = the chat provider), built on first use, so importing this
    module never needs an API key (tests stay offline)."""
    p = PROVIDERS[name or provider_name()]
    key = os.environ.get(p.key_env)
    if not key:
        raise ValueError(f"{p.key_env} is empty: add it to .env")
    return OpenAI(api_key=key, base_url=p.base_url)


class NoCreditError(RuntimeError):
    """OpenAI's 429 insufficient_quota: the organization's prepaid credit is used up. Waiting won't help."""


def _no_credit(e: RateLimitError) -> NoCreditError | None:
    if "insufficient_quota" not in str(e) and "credit_balance" not in str(e):
        return None
    return NoCreditError("the OpenAI organization has no credit left (429 insufficient_quota): add credit in the "
                         "OpenAI dashboard (Settings -> Billing); even the free daily data-sharing tokens need a "
                         "positive balance")


def _complete(
    messages: list[Message],
    model: str | None,
    step: str,
    ledger_path: Path,
    temperature: float | None = None,
    **request: object,
) -> ChatResult:
    """The one place that sends a request: every call, plain or JSON, is priced and recorded here."""
    model = model or model_for(step)
    ledger.cost_usd(model, 0, 0)  # fail before paying if this model has no price
    reasoning = provider_of(model) == "openai" and model.startswith(_REASONING_FAMILIES)
    if temperature is not None:  # not `if temperature:`, because 0.0 is a real choice
        if reasoning:  # OpenAI's reasoning models refuse any temperature but 1: say so before sending anything
            raise ValueError(f"{model} is a reasoning model and takes no temperature: use a model like gpt-4.1-mini "
                             "for a temperature experiment")
        request["temperature"] = temperature  # left out = the provider's default (1.0 for Gemini 3)
    if (effort := reasoning_effort(model)) is not None:
        request.setdefault("reasoning_effort", effort)

    start = time.perf_counter()
    try:
        response = _client(provider_of(model)).chat.completions.create(model=model, messages=messages, **request)
    except RateLimitError as e:
        if (no_credit := _no_credit(e)) is not None:
            raise no_credit from e
        raise
    latency = time.perf_counter() - start

    usage = response.usage
    cached = (usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0) or 0
    # Gemini 3 Flash models think before answering: e.g. 5,444 in + 123 out but total_tokens 6,184
    # (Phase 6 build notes, E13). The gap is billed as output, so it is priced as output here.
    # OpenAI's reasoning models count their thinking INSIDE completion_tokens (completion_tokens_details
    # says how much): total = prompt + completion, the gap is 0 and nothing is counted twice.
    reasoning = max(0, (getattr(usage, "total_tokens", 0) or 0) - usage.prompt_tokens - usage.completion_tokens)
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
        reasoning_tokens=reasoning,
        cost_usd=ledger.cost_usd(answered_by, usage.prompt_tokens, usage.completion_tokens + reasoning, cached),
        latency_s=round(latency, 2),
    )
    ledger.record(
        ledger.LedgerEntry(
            step=step,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            cached_tokens=result.cached_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
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
    result = chat_schema(messages, schema, model, step, ledger_path, temperature)
    return schema.model_validate_json(result.text), result


def chat_schema(
    messages: list[Message],
    schema: type[BaseModel],
    model: str | None = None,
    step: str = "adhoc",
    ledger_path: Path = DEFAULT_LEDGER,
    temperature: float | None = None,
) -> ChatResult:
    """chat_json() without the validation: JSON in `schema`'s shape, recorded, returned as text.

    For answers that hold several items (the Writer's 3 script options): one option breaking a
    rule must not throw away the other two, so the caller validates item by item.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
    }
    return _complete(messages, model, step, ledger_path, temperature, response_format=response_format)


# --- pacing for the free tier ---------------------------------------------------------------

MIN_INTERVAL_S = 4.5  # Gemini's free tier allows ~15 requests a minute: one every 4 s, plus a margin
CHAT_RETRIES = 4
OVERLOADED_WAIT_S = 30  # x attempt: a 503 "model is currently experiencing high demand" passes in a minute or two
_last_request = float("-inf")  # time.monotonic() of the last paced request, shared by every caller


class DailyQuotaError(RuntimeError):
    """A per-day quota is used up: waiting minutes won't help, so it isn't retried (Phase 6, E11)."""


def min_interval() -> float:
    """Seconds between paced requests: MIN_INTERVAL_S on Gemini's free tier; none on OpenAI, whose
    tier-1 limits (hundreds of requests a minute) a ~10-call run never comes near."""
    return MIN_INTERVAL_S if provider_name() == "gemini" else 0.0


def paced[T](call: Callable[[], T], sleep: Callable[[float], None] = time.sleep) -> T:
    """Run one LLM request at most every min_interval(), and wait out a 429 or a 5xx instead of failing.

    A 429's wait is the one the error names ("Please retry in 12.75s", see _retry_delay); a 5xx
    (Gemini's 503 "high demand", which ended run 1 of the Phase 6 build) waits 30 s, 60 s, 90 s: the
    SDK's own two quick retries are not enough for a demand spike. A refused request isn't billed,
    so only the call that succeeds lands in the ledger. The graph's nodes (Phase 6) send every
    request through here. OpenAI's "no credit" 429 is never retried: _complete() raises NoCreditError.
    """
    global _last_request
    for attempt in range(1, CHAT_RETRIES + 1):
        wait = _last_request + min_interval() - time.monotonic()
        if wait > 0:
            sleep(wait)
        _last_request = time.monotonic()
        try:
            return call()
        except RateLimitError as e:
            if "PerDay" in str(e):  # a daily quota (20 requests per model on the free tier) won't clear in a minute
                raise DailyQuotaError(f"the free tier's daily request quota for this model is used up (it resets at "
                                      f"midnight Pacific time); try another model: {str(e)[:300]}") from e
            if attempt == CHAT_RETRIES:
                raise
            delay = _retry_delay(e, attempt)
            log.warning("chat: 429 rate limit, waiting %.0f s (attempt %d of %d): %s", delay, attempt, CHAT_RETRIES,
                        str(e)[:300])
            sleep(delay)
        except InternalServerError as e:
            if attempt == CHAT_RETRIES:
                raise
            log.warning("chat: %d from the provider, waiting %d s (attempt %d of %d)", e.status_code,
                        OVERLOADED_WAIT_S * attempt, attempt, CHAT_RETRIES)
            sleep(OVERLOADED_WAIT_S * attempt)
    raise AssertionError("unreachable")


# --- embeddings ------------------------------------------------------------------------------

EMBED_DIMENSIONS = 768  # every model we use can shorten its vectors (3072 / 1536 by default); 768 is
# one of Google's recommended sizes: 4x less to store and compare, little lost for short texts.
# OpenAI's text-embedding-3 models shorten too (the `dimensions` parameter, "Matryoshka" training).
EMBED_BATCH = 100  # texts per request: Gemini's batch limit
# Gemini's free tier counts every TEXT as one request, 100 per minute per model, batched or not
# (Phase 5 build notes): re-indexing twice in a minute hits it. A 429 waits as long as the
# error says (it names the delay) and tries again.
EMBED_RETRIES = 3
TIKTOKEN_ENCODING = "o200k_base"  # only for estimating tokens a provider didn't report

log = logging.getLogger("shortsmith.llm")  # stderr, never stdout: the MCP server's stdout is its channel


class EmbedResult(BaseModel):
    """Vectors for a list of texts, in the same order, and what they cost."""

    vectors: list[list[float]] = Field(repr=False)
    model: str
    dimensions: int
    prompt_tokens: int  # embeddings only have input tokens
    tokens_estimated: bool  # True = counted with tiktoken, because the provider sent no usage
    cost_usd: float  # list price
    latency_s: float


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TIKTOKEN_ENCODING)  # loaded once, on first use


def estimate_tokens(text: str) -> int:
    """tiktoken's count: exact for OpenAI models, an estimate for Gemini (~6% high on our tests)."""
    return len(_encoding().encode(text))


def _retry_delay(error: RateLimitError, attempt: int) -> float:
    """How long a 429 asks us to wait: Gemini says "Please retry in 12.75s"; else 20 s per attempt."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) + 1 if match else 20.0 * attempt


def _embed_request(batch: list[str], model: str, dimensions: int) -> Any:
    """One embeddings request, retried after a 429. A refused request isn't billed."""
    for attempt in range(1, EMBED_RETRIES + 1):
        try:
            return _client(provider_of(model)).embeddings.create(model=model, input=batch, dimensions=dimensions)
        except RateLimitError as e:
            if (no_credit := _no_credit(e)) is not None:
                raise no_credit from e
            if attempt == EMBED_RETRIES:
                raise
            delay = _retry_delay(e, attempt)
            log.warning("embeddings: 429 rate limit (%d texts), waiting %.0f s", len(batch), delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def embed(
    texts: list[str],
    model: str | None = None,
    dimensions: int = EMBED_DIMENSIONS,
    step: str = "embed",
    ledger_path: Path = DEFAULT_LEDGER,
) -> EmbedResult:
    """Turn texts into vectors, up to EMBED_BATCH per request; every request goes in the ledger.

    Gemini's OpenAI-compatible endpoint answers with usage: null for embeddings (its native
    endpoint does count them), so those tokens are estimated with tiktoken and marked as such.
    The model goes to its own provider (provider_of), whatever LLM_PROVIDER says.
    """
    model = model or embedding_model()
    ledger.cost_usd(model, 0, 0)  # no price, no call
    vectors: list[list[float]] = []
    tokens, estimated, latency, answered_by = 0, False, 0.0, model
    for first in range(0, len(texts), EMBED_BATCH):
        batch = texts[first : first + EMBED_BATCH]
        start = time.perf_counter()
        response = _embed_request(batch, model, dimensions)
        latency += time.perf_counter() - start
        answered_by = response.model.removeprefix("models/")
        if response.usage is not None and response.usage.prompt_tokens:
            batch_tokens, batch_estimated = response.usage.prompt_tokens, False
        else:
            batch_tokens, batch_estimated = sum(estimate_tokens(text) for text in batch), True
        tokens, estimated = tokens + batch_tokens, estimated or batch_estimated
        ledger.record(  # first, before anything else can fail: the request is already paid for
            ledger.LedgerEntry(step=step, model=answered_by, prompt_tokens=batch_tokens, completion_tokens=0,
                               cost_usd=ledger.cost_usd(answered_by, batch_tokens, 0), tokens_estimated=batch_estimated),
            ledger_path,
        )
        items = response.data
        if all(item.index is not None for item in items):  # OpenAI numbers them; Gemini sends index: null, in order
            items = sorted(items, key=lambda item: item.index)
        vectors += [item.embedding for item in items]
    if len(vectors) != len(texts) or any(len(v) != dimensions for v in vectors):
        raise ValueError(f"asked for {len(texts)} vectors of {dimensions} numbers, got "
                         f"{len(vectors)} of {sorted({len(v) for v in vectors})}")
    return EmbedResult(vectors=vectors, model=answered_by, dimensions=dimensions, prompt_tokens=tokens,
                       tokens_estimated=estimated, cost_usd=ledger.cost_usd(answered_by, tokens, 0),
                       latency_s=round(latency, 2))


# --- the writer ------------------------------------------------------------------------------

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


def writer_message(topic: str, language: str, research: ResearchNotes | None = None) -> str:
    """The writer's user message. Research notes, when there are any, go here and not into the
    system prompt: they change with every topic, the rules don't."""
    message = f"Topic: {topic}\nLanguage: {LANGUAGES[language]} (language code '{language}')"
    return f"{message}\n\n{research.for_writer()}" if research else message


def write_script(
    topic: str, language: str = "en", model: str | None = None, research: ResearchNotes | None = None
) -> tuple[Script, ChatResult]:
    """One topic in, one validated Script out (this becomes the Writer agent in Phase 6).
    With `research` (agents/researcher.py), the script is grounded in real trends and hook patterns."""
    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": writer_message(topic, language, research)},
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
