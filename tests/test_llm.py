"""chat() with a fake OpenAI client: checks the ledger wiring without any API call or key."""

import json
from types import SimpleNamespace as NS

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, ValidationError

from shortsmith import ledger, llm
from shortsmith.ledger import LedgerEntry
from shortsmith.models import HookIdea, ResearchNotes

MESSAGES = [{"role": "user", "content": "hi"}]


def _fake_client(response, seen: dict | None = None):
    """Mimics the one method chat() uses: client.chat.completions.create(**kwargs).

    If `seen` is given, the request's keyword arguments are copied into it.
    """

    def create(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        return response

    return NS(chat=NS(completions=NS(create=create)))


def _message(content: str | None) -> ChatCompletionMessage:
    """The SDK's real message type: _complete() calls its model_dump()."""
    return ChatCompletionMessage(role="assistant", content=content)


def _response(content: str, model: str = "gpt-4.1-mini-2025-04-14"):
    return NS(
        model=model,
        choices=[NS(message=_message(content))],
        usage=NS(prompt_tokens=100, completion_tokens=50, prompt_tokens_details=None),
    )


class Point(BaseModel):
    x: int
    y: int


def test_chat_returns_usage_and_records_it(monkeypatch, tmp_path):
    response = NS(
        model="gpt-4.1-mini-2025-04-14",
        choices=[NS(message=_message("Hello!"))],
        usage=NS(prompt_tokens=1000, completion_tokens=500, prompt_tokens_details=NS(cached_tokens=200)),
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")  # tests must not depend on your .env
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(response))
    path = tmp_path / "ledger.jsonl"

    result = llm.chat(MESSAGES, step="writer", ledger_path=path)

    # 800 x $0.40 + 200 x $0.10 + 500 x $1.60, per 1M tokens
    assert result.text == "Hello!"
    assert result.cost_usd == pytest.approx(0.00114)
    entry = LedgerEntry.model_validate_json(path.read_text(encoding="utf-8"))
    assert (entry.step, entry.prompt_tokens, entry.cached_tokens) == ("writer", 1000, 200)


def test_unpriced_model_fails_before_any_paid_call(monkeypatch, tmp_path):
    def must_not_be_called():
        raise AssertionError("the API was called for a model without a price")

    monkeypatch.setattr(llm, "_client", must_not_be_called)
    with pytest.raises(ValueError):
        llm.chat(MESSAGES, model="mystery-model", ledger_path=tmp_path / "ledger.jsonl")


def test_provider_is_chosen_by_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert "generativelanguage.googleapis.com" in llm.provider().base_url
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        llm.provider()


def test_gemini_model_prefix_is_stripped_before_pricing(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    response = NS(
        model="models/gemini-3.5-flash-lite",
        choices=[NS(message=_message("Salom!"))],
        usage=NS(prompt_tokens=1_000_000, completion_tokens=0, prompt_tokens_details=None),
    )
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(response))

    result = llm.chat(MESSAGES, ledger_path=tmp_path / "ledger.jsonl")

    assert result.model == "gemini-3.5-flash-lite"
    assert result.cost_usd == pytest.approx(0.30)


def test_temperature_is_sent_only_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    seen: dict = {}
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(_response("hi"), seen))

    llm.chat(MESSAGES, temperature=0.0, ledger_path=tmp_path / "ledger.jsonl")
    assert seen["temperature"] == 0.0  # 0.0 is falsy: a sloppy `if temperature:` would drop it

    seen.clear()
    llm.chat(MESSAGES, ledger_path=tmp_path / "ledger.jsonl")
    assert "temperature" not in seen  # not set = the provider's default


def test_chat_json_sends_the_schema_and_parses_the_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    seen: dict = {}
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(_response('{"x": 1, "y": 2}'), seen))

    point, result = llm.chat_json(MESSAGES, Point, ledger_path=tmp_path / "ledger.jsonl")

    assert point == Point(x=1, y=2)
    sent = seen["response_format"]["json_schema"]
    assert (sent["name"], sent["strict"]) == ("Point", True)
    assert set(sent["schema"]["properties"]) == {"x", "y"}


def test_chat_json_records_the_call_even_when_validation_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(_response('{"x": "oops"}')))
    path = tmp_path / "ledger.jsonl"

    with pytest.raises(ValidationError):
        llm.chat_json(MESSAGES, Point, ledger_path=path)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1  # paid, so recorded


def test_chat_tools_sends_the_schemas_and_keeps_the_message_as_returned(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    message = ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_80370", "type": "function",
            "function": {"name": "generate_image", "arguments": '{"scene":1}'},
            "extra_content": {"google": {"thought_signature": "EmAK..."}},  # a field the SDK doesn't know
        }],
    })
    response = NS(model="gemini-3.5-flash-lite", choices=[NS(message=message)],
                  usage=NS(prompt_tokens=287, completion_tokens=28, prompt_tokens_details=None))
    seen: dict = {}
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(response, seen))
    schemas = [{"type": "function", "function": {"name": "generate_image", "parameters": {"type": "object"}}}]
    path = tmp_path / "ledger.jsonl"

    result = llm.chat_tools([{"role": "user", "content": "go"}], schemas, ledger_path=path)

    assert seen["tools"] is schemas
    assert result.text == ""
    assert result.tool_calls == [llm.ToolCall(id="call_80370", name="generate_image", arguments='{"scene":1}')]
    assert result.message["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "EmAK..."}}
    assert "content" not in result.message  # None fields are left out, not sent back as null
    assert LedgerEntry.model_validate_json(path.read_text(encoding="utf-8")).step == "agent"


# --- embed() -------------------------------------------------------------------------------------


def _embedding_client(batches: list, seen: list | None = None):
    """Mimics client.embeddings.create(**kwargs): answers with the next prepared response."""
    answers = iter(batches)

    def create(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return NS(embeddings=NS(create=create))


def _embeddings(vectors, model="gemini-embedding-2", usage=None, indexed=False):
    return NS(model=model, usage=usage,
              data=[NS(embedding=v, index=i if indexed else None) for i, v in enumerate(vectors)])


def test_embed_estimates_tokens_when_the_provider_sends_no_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    seen: list = []
    monkeypatch.setattr(llm, "_client", lambda: _embedding_client([_embeddings([[0.1, 0.2], [0.3, 0.4]])], seen))
    path = tmp_path / "ledger.jsonl"

    result = llm.embed(["hello world", "salom dunyo"], dimensions=2, ledger_path=path)

    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]  # Gemini sends index: null, in input order
    assert seen[0] == {"model": "gemini-embedding-2", "input": ["hello world", "salom dunyo"], "dimensions": 2}
    tokens = llm.estimate_tokens("hello world") + llm.estimate_tokens("salom dunyo")
    assert (result.prompt_tokens, result.tokens_estimated) == (tokens, True)
    entry = LedgerEntry.model_validate_json(path.read_text(encoding="utf-8"))
    assert (entry.step, entry.prompt_tokens, entry.completion_tokens, entry.tokens_estimated) == ("embed", tokens, 0, True)
    assert entry.cost_usd == pytest.approx(tokens * 0.20 / 1e6)


def test_embed_uses_the_providers_usage_and_index_when_it_has_them(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    shuffled = _embeddings([[1.0], [2.0]], model="text-embedding-3-small", usage=NS(prompt_tokens=7), indexed=True)
    shuffled.data.reverse()  # OpenAI numbers the vectors; the order of the list doesn't matter then
    monkeypatch.setattr(llm, "_client", lambda: _embedding_client([shuffled]))

    result = llm.embed(["a", "b"], dimensions=1, ledger_path=tmp_path / "ledger.jsonl")

    assert result.vectors == [[1.0], [2.0]]
    assert (result.model, result.prompt_tokens, result.tokens_estimated) == ("text-embedding-3-small", 7, False)


def test_embed_splits_big_inputs_into_batches_and_records_each_request(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "EMBED_BATCH", 2)
    seen: list = []
    client = _embedding_client([_embeddings([[1.0], [2.0]]), _embeddings([[3.0]])], seen)  # one client: one queue of answers
    monkeypatch.setattr(llm, "_client", lambda: client)
    path = tmp_path / "ledger.jsonl"

    result = llm.embed(["a", "b", "c"], dimensions=1, ledger_path=path)

    assert [call["input"] for call in seen] == [["a", "b"], ["c"]]
    assert result.vectors == [[1.0], [2.0], [3.0]]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2  # one ledger line per paid request


def test_embed_rejects_vectors_of_the_wrong_size(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "_client", lambda: _embedding_client([_embeddings([[1.0, 2.0, 3.0]])]))
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(ValueError, match="asked for 1 vectors of 2 numbers"):
        llm.embed(["a"], dimensions=2, ledger_path=path)
    assert path.exists()  # the request was answered, so it is in the ledger anyway


def test_embed_waits_as_long_as_a_429_says_and_tries_again(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    request = httpx.Request("POST", "https://example.test/embeddings")
    limited = openai.RateLimitError(
        "Quota exceeded for metric: embed_content_free_tier_requests, limit: 100. Please retry in 12.75s.",
        response=httpx.Response(429, request=request), body=None)
    client = _embedding_client([limited, _embeddings([[1.0]])])
    monkeypatch.setattr(llm, "_client", lambda: client)
    waited: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", waited.append)

    result = llm.embed(["a"], dimensions=1, ledger_path=tmp_path / "ledger.jsonl")

    assert result.vectors == [[1.0]] and waited == [pytest.approx(13.75)]
    assert len((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 1  # a refused request isn't billed


def test_embed_refuses_an_unpriced_model_before_any_call(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_client", lambda: pytest.fail("called the API for a model without a price"))
    with pytest.raises(ValueError, match="No price"):
        llm.embed(["a"], model="mystery-embedding", ledger_path=tmp_path / "ledger.jsonl")


def test_every_provider_has_a_priced_embedding_model():
    for p in llm.PROVIDERS.values():
        assert ledger.cost_usd(p.embedding_model, 1_000_000, 0) > 0


# --- the writer with research ------------------------------------------------------------------


NOTES = ResearchNotes(
    trending=["Cafe tours with prices get 30k+ views"],
    hooks=[HookIdea(pattern="Hidden gem", hook="Tashkent's best cafe has no sign", why="locals love secrets")],
    angles=["steam on a cold morning"],
    avoid=["luxury"],
    visual_style="warm amber light, 35mm film look",
)


def test_the_writer_message_without_research_is_unchanged():
    assert llm.writer_message("tea", "uz") == "Topic: tea\nLanguage: Uzbek (Latin script) (language code 'uz')"


def test_write_script_puts_the_research_notes_into_the_user_message(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    seen: dict = {}
    scenes = [{"idx": i, "narration": " ".join(["word"] * 13), "visual_prompt": "p", "duration_s": 5, "motion": "zoom_in"}
              for i in range(1, 7)]
    answer = json.dumps({"title": "t", "hook": "h", "scenes": scenes, "cta": "c", "language": "en"})
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(_response(answer, "gemini-3.5-flash-lite"), seen))
    monkeypatch.setattr(llm, "DEFAULT_LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(llm.ledger, "record", lambda entry, path: None)

    llm.write_script("cafe", "en", research=NOTES)

    system, user = seen["messages"]
    assert system["content"] == llm.WRITER_SYSTEM_PROMPT  # the rules don't change with research
    assert user["content"].startswith("Topic: cafe\nLanguage: English")
    assert '- Hidden gem: "Tashkent\'s best cafe has no sign" (locals love secrets)' in user["content"]
    assert "End every visual_prompt with these style words: warm amber light, 35mm film look" in user["content"]


# --- Phase 6: chat_schema and paced ------------------------------------------------------------------


def test_chat_schema_sends_the_schema_and_leaves_validation_to_the_caller(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(_response('{"x": "not a number"}'), seen))
    result = llm.chat_schema(MESSAGES, Point, step="writer", ledger_path=tmp_path / "ledger.jsonl")
    assert result.text == '{"x": "not a number"}'  # chat_json would raise here; the caller decides
    assert seen["response_format"]["json_schema"]["name"] == "Point"
    assert (tmp_path / "ledger.jsonl").exists()


def test_paced_spaces_requests_and_waits_out_a_429(monkeypatch):
    request = httpx.Request("POST", "https://example.test/chat/completions")
    limited = openai.RateLimitError("429 RESOURCE_EXHAUSTED. Please retry in 12.75s.",
                                    response=httpx.Response(429, request=request), body=None)
    answers = iter([limited, "ok"])
    waits: list[float] = []

    def call():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(llm, "_last_request", float("-inf"))
    assert llm.paced(call, sleep=waits.append) == "ok"
    assert waits[0] == 13.75  # the delay the error named, + 1 s
    assert 0 < waits[1] <= llm.MIN_INTERVAL_S  # and the next request still keeps the spacing


def test_paced_gives_up_after_the_last_retry(monkeypatch):
    request = httpx.Request("POST", "https://example.test/chat/completions")

    def call():
        raise openai.RateLimitError("429", response=httpx.Response(429, request=request), body=None)

    with pytest.raises(openai.RateLimitError):
        llm.paced(call, sleep=lambda s: None)


def test_paced_waits_out_a_503_high_demand(monkeypatch):
    request = httpx.Request("POST", "https://example.test/chat/completions")
    overloaded = openai.InternalServerError("This model is currently experiencing high demand.",
                                            response=httpx.Response(503, request=request), body=None)
    answers = iter([overloaded, overloaded, "ok"])
    waits: list[float] = []

    def call():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(llm, "MIN_INTERVAL_S", 0)
    assert llm.paced(call, sleep=waits.append) == "ok"
    assert waits == [30, 60]


def test_thinking_tokens_hidden_in_total_tokens_are_priced_as_output(monkeypatch, tmp_path):
    """Gemini's compatible endpoint: completion_tokens leaves the model's thinking out; total_tokens has it."""
    response = NS(model="gemini-3.6-flash", choices=[NS(message=_message("{}"))],
                  usage=NS(prompt_tokens=5444, completion_tokens=123, total_tokens=6184, prompt_tokens_details=None))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(response))
    result = llm.chat(MESSAGES, step="critic", ledger_path=tmp_path / "ledger.jsonl")

    assert result.reasoning_tokens == 617 and result.total_tokens == 6184
    assert result.cost_usd == pytest.approx((5444 * 0.75 + (123 + 617) * 3.75) / 1e6)  # thinking at the output rate
    entry = LedgerEntry.model_validate_json((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert (entry.completion_tokens, entry.reasoning_tokens) == (123, 617)


def test_openai_counts_reasoning_inside_completion_tokens_so_nothing_is_added(monkeypatch, tmp_path):
    response = NS(model="gpt-4.1-mini", choices=[NS(message=_message("hi"))],
                  usage=NS(prompt_tokens=100, completion_tokens=50, total_tokens=150, prompt_tokens_details=None))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(response))
    assert llm.chat(MESSAGES, ledger_path=tmp_path / "ledger.jsonl").reasoning_tokens == 0


def test_a_daily_quota_429_is_not_retried():
    request = httpx.Request("POST", "https://example.test/chat/completions")
    calls: list[int] = []

    def call():
        calls.append(1)
        raise openai.RateLimitError("429 ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' ... "
                                    "Please retry in 11h41m32.4s.", response=httpx.Response(429, request=request), body=None)

    with pytest.raises(llm.DailyQuotaError, match="daily request quota"):
        llm.paced(call, sleep=lambda s: None)
    assert len(calls) == 1
