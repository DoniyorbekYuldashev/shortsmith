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
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(response))
    path = tmp_path / "ledger.jsonl"

    result = llm.chat(MESSAGES, step="writer", ledger_path=path)

    # 800 x $0.40 + 200 x $0.10 + 500 x $1.60, per 1M tokens
    assert result.text == "Hello!"
    assert result.cost_usd == pytest.approx(0.00114)
    entry = LedgerEntry.model_validate_json(path.read_text(encoding="utf-8"))
    assert (entry.step, entry.prompt_tokens, entry.cached_tokens) == ("writer", 1000, 200)


def test_unpriced_model_fails_before_any_paid_call(monkeypatch, tmp_path):
    def must_not_be_called(*_):
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
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(response))

    result = llm.chat(MESSAGES, ledger_path=tmp_path / "ledger.jsonl")

    assert result.model == "gemini-3.5-flash-lite"
    assert result.cost_usd == pytest.approx(0.30)


def test_temperature_is_sent_only_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    seen: dict = {}
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(_response("hi"), seen))

    llm.chat(MESSAGES, model="gpt-4.1-mini", temperature=0.0, ledger_path=tmp_path / "ledger.jsonl")
    assert seen["temperature"] == 0.0  # 0.0 is falsy: a sloppy `if temperature:` would drop it

    seen.clear()
    llm.chat(MESSAGES, model="gpt-4.1-mini", ledger_path=tmp_path / "ledger.jsonl")
    assert "temperature" not in seen  # not set = the provider's default


def test_chat_json_sends_the_schema_and_parses_the_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    seen: dict = {}
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(_response('{"x": 1, "y": 2}'), seen))

    point, result = llm.chat_json(MESSAGES, Point, ledger_path=tmp_path / "ledger.jsonl")

    assert point == Point(x=1, y=2)
    sent = seen["response_format"]["json_schema"]
    assert (sent["name"], sent["strict"]) == ("Point", True)
    assert set(sent["schema"]["properties"]) == {"x", "y"}


def test_chat_json_records_the_call_even_when_validation_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(_response('{"x": "oops"}')))
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
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(response, seen))
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
    monkeypatch.setattr(llm, "_client", lambda *_: _embedding_client([_embeddings([[0.1, 0.2], [0.3, 0.4]])], seen))
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
    monkeypatch.setattr(llm, "_client", lambda *_: _embedding_client([shuffled]))

    result = llm.embed(["a", "b"], dimensions=1, ledger_path=tmp_path / "ledger.jsonl")

    assert result.vectors == [[1.0], [2.0]]
    assert (result.model, result.prompt_tokens, result.tokens_estimated) == ("text-embedding-3-small", 7, False)


def test_embed_splits_big_inputs_into_batches_and_records_each_request(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "EMBED_BATCH", 2)
    seen: list = []
    client = _embedding_client([_embeddings([[1.0], [2.0]]), _embeddings([[3.0]])], seen)  # one client: one queue of answers
    monkeypatch.setattr(llm, "_client", lambda *_: client)
    path = tmp_path / "ledger.jsonl"

    result = llm.embed(["a", "b", "c"], dimensions=1, ledger_path=path)

    assert [call["input"] for call in seen] == [["a", "b"], ["c"]]
    assert result.vectors == [[1.0], [2.0], [3.0]]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2  # one ledger line per paid request


def test_embed_rejects_vectors_of_the_wrong_size(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "_client", lambda *_: _embedding_client([_embeddings([[1.0, 2.0, 3.0]])]))
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
    monkeypatch.setattr(llm, "_client", lambda *_: client)
    waited: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", waited.append)

    result = llm.embed(["a"], dimensions=1, ledger_path=tmp_path / "ledger.jsonl")

    assert result.vectors == [[1.0]] and waited == [pytest.approx(13.75)]
    assert len((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 1  # a refused request isn't billed


def test_embed_refuses_an_unpriced_model_before_any_call(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_client", lambda *_: pytest.fail("called the API for a model without a price"))
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
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(_response(answer, "gemini-3.5-flash-lite"), seen))
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
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(_response('{"x": "not a number"}'), seen))
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

    monkeypatch.setenv("LLM_PROVIDER", "gemini")  # the spacing is Gemini's free tier's; OpenAI needs none
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
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(response))
    result = llm.chat(MESSAGES, step="critic", ledger_path=tmp_path / "ledger.jsonl")

    assert result.reasoning_tokens == 617 and result.total_tokens == 6184
    assert result.cost_usd == pytest.approx((5444 * 0.75 + (123 + 617) * 3.75) / 1e6)  # thinking at the output rate
    entry = LedgerEntry.model_validate_json((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert (entry.completion_tokens, entry.reasoning_tokens) == (123, 617)


def test_openai_counts_reasoning_inside_completion_tokens_so_nothing_is_added(monkeypatch, tmp_path):
    response = NS(model="gpt-4.1-mini", choices=[NS(message=_message("hi"))],
                  usage=NS(prompt_tokens=100, completion_tokens=50, total_tokens=150, prompt_tokens_details=None))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(response))
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


# --- providers per role, OpenAI's request rules (the OpenAI switch, 2026-10-26) --------------------------------


def test_each_role_uses_the_providers_default_unless_set_apart(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert [llm.model_for(step) for step in ("researcher", "writer", "writer-pick", "director", "revise")] == ["gpt-5-mini"] * 5
    assert llm.model_for("critic") == "gpt-5.1" and llm.model_for("adhoc") == "gpt-5-mini"
    monkeypatch.setenv("SHORTSMITH_WRITER_MODEL", "gpt-4.1-mini")
    assert (llm.model_for("writer"), llm.model_for("writer-pick"), llm.model_for("director")) == (
        "gpt-4.1-mini", "gpt-4.1-mini", "gpt-5-mini")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert (llm.model_for("director"), llm.model_for("critic")) == ("gemini-3.5-flash-lite", "gemini-3.8-flash")


def test_a_model_goes_to_its_own_providers_client(monkeypatch, tmp_path):
    """A Gemini critic while LLM_PROVIDER=openai: the request must reach Google's address with Google's key."""
    response = _response("{}", model="gemini-3.8-flash")
    used: list[str | None] = []

    def client(name=None):
        used.append(name)
        return _fake_client(response)

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("SHORTSMITH_CRITIC_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(llm, "_client", client)
    llm.chat(MESSAGES, step="critic", ledger_path=tmp_path / "ledger.jsonl")
    assert used == ["gemini"] and llm.provider_of("models/gemini-embedding-2") == "gemini"
    assert llm.provider_of("gpt-5-mini") == llm.provider_of("text-embedding-3-small") == "openai"


def test_reasoning_effort_goes_only_to_openais_reasoning_models(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(_response("hi", "gpt-5-mini-2025-08-07"), seen))
    llm.chat(MESSAGES, model="gpt-5-mini", ledger_path=tmp_path / "ledger.jsonl")
    assert seen["reasoning_effort"] == "low"
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")  # empty = the API's own default: nothing sent
    seen.clear()
    llm.chat(MESSAGES, model="gpt-5-mini", ledger_path=tmp_path / "ledger.jsonl")
    assert "reasoning_effort" not in seen
    assert llm.reasoning_effort("gpt-4.1-mini") is None and llm.reasoning_effort("gemini-3.8-flash") is None
    assert llm.reasoning_effort("gpt-5.1-chat-latest") is None


def test_openai_reasoning_inside_completion_tokens_is_priced_once(monkeypatch, tmp_path):
    """gpt-5-mini: 300 of the 500 completion tokens are reasoning, and total = prompt + completion."""
    response = NS(model="gpt-5-mini-2025-08-07", choices=[NS(message=_message("{}"))],
                  usage=NS(prompt_tokens=1000, completion_tokens=500, total_tokens=1500, prompt_tokens_details=None,
                           completion_tokens_details=NS(reasoning_tokens=300)))
    monkeypatch.setattr(llm, "_client", lambda *_: _fake_client(response))
    result = llm.chat(MESSAGES, model="gpt-5-mini", ledger_path=tmp_path / "ledger.jsonl")
    assert result.reasoning_tokens == 0 and result.total_tokens == 1500
    assert result.cost_usd == pytest.approx((1000 * 0.25 + 500 * 2.00) / 1e6)


def test_no_credit_is_a_clear_error_and_never_retried(monkeypatch, tmp_path):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    calls: list[int] = []

    def create(**kwargs):
        calls.append(1)
        raise openai.RateLimitError("Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
                                    response=httpx.Response(429, request=request), body=None)

    monkeypatch.setenv("LLM_PROVIDER", "openai")  # no spacing: any sleep here would be a retry
    monkeypatch.setattr(llm, "_client", lambda *_: NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(llm.NoCreditError, match="no credit left"):
        llm.paced(lambda: llm.chat(MESSAGES, model="gpt-5-mini", ledger_path=tmp_path / "ledger.jsonl"),
                  sleep=lambda s: pytest.fail("waited for a 429 that waiting can't fix"))
    assert len(calls) == 1 and not (tmp_path / "ledger.jsonl").exists()


def test_openai_needs_no_spacing_and_gemini_does(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert llm.min_interval() == 0
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm.min_interval() == llm.MIN_INTERVAL_S


def _strict_problems(node, path: str, problems: list[str]) -> None:
    """OpenAI's strict structured outputs: every object lists all its keys as required and forbids extra keys;
    no allOf/not/if/then/else; and (from the SDK's own strict converter) no `default`."""
    if isinstance(node, dict):
        if "properties" in node:
            optional = set(node["properties"]) - set(node.get("required", []))
            if optional:
                problems.append(f"{path}: optional keys {sorted(optional)}")
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties is not false")
        problems += [f"{path}: {key}" for key in ("allOf", "not", "if", "then", "else", "default") if key in node]
        for key, value in node.items():
            _strict_problems(value, f"{path}/{key}", problems)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _strict_problems(value, f"{path}[{i}]", problems)


def test_every_schema_we_send_meets_openais_strict_rules():
    from shortsmith.agents.director import DirectorPlan
    from shortsmith.agents.reviser import SceneFixes
    from shortsmith.agents.writer import ScriptOptions, WriterPick
    from shortsmith.models import CriticVerdict, Script

    for schema in (ResearchNotes, Script, ScriptOptions, WriterPick, DirectorPlan, CriticVerdict, SceneFixes):
        problems: list[str] = []
        root = schema.model_json_schema()
        _strict_problems(root, schema.__name__, problems)
        assert root["type"] == "object" and "anyOf" not in root
        assert problems == [], problems


def test_the_embedding_provider_is_independent_of_the_chat_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert llm.embedding_model() == llm.PROVIDERS[llm.EMBEDDING_PROVIDER_DEFAULT].embedding_model
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    assert llm.embedding_model() == "gemini-embedding-2"  # chat on OpenAI, the index on Gemini's vectors
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    assert llm.embedding_model() == "text-embedding-3-small"
    monkeypatch.setenv("SHORTSMITH_EMBEDDING_MODEL", "text-embedding-3-large")
    assert llm.embedding_model() == "text-embedding-3-large"
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nope")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        llm.embedding_provider()

    used: list[str | None] = []
    batch = _embeddings([[0.5] * 4], model="text-embedding-3-small", usage=NS(prompt_tokens=7), indexed=True)

    def client(name=None):
        used.append(name)
        return _embedding_client([batch])

    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")  # chat on Gemini, embeddings on OpenAI: still OpenAI's address
    result = llm.embed(["a"], model="text-embedding-3-small", dimensions=4, ledger_path=tmp_path / "ledger.jsonl")
    assert used == ["openai"] and result.prompt_tokens == 7 and result.cost_usd == pytest.approx(7 * 0.02 / 1e6)


def test_the_rag_tables_are_one_per_model_and_size():
    from shortsmith.research import rag, stores

    names = {stores.table_name(rag.collection_name(m, 768)) for m in ("gemini-embedding-2", "text-embedding-3-small")}
    assert names == {"kb_gemini_embedding_2_768", "kb_text_embedding_3_small_768"}


def test_a_dated_snapshot_has_its_models_price_and_a_longer_name_has_none():
    assert ledger.base_model("gpt-5-mini-2025-08-07") == "gpt-5-mini"
    assert ledger.cost_usd("gpt-5.1-2025-11-13", 1_000_000, 0) == pytest.approx(1.25)
    for unknown in ("gpt-5.5", "gpt-5-pro", "gpt-5.1-codex-max"):  # "starts with gpt-5" must not price them as gpt-5
        with pytest.raises(ValueError, match="No price"):
            ledger.cost_usd(unknown, 1, 1)


def test_a_temperature_for_an_openai_reasoning_model_fails_before_sending(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_client", lambda *_: pytest.fail("sent a request OpenAI would refuse"))
    with pytest.raises(ValueError, match="takes no temperature"):
        llm.chat(MESSAGES, model="gpt-5-mini", temperature=0.0, ledger_path=tmp_path / "ledger.jsonl")
