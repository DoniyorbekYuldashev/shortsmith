"""chat() with a fake OpenAI client: checks the ledger wiring without any API call or key."""

from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, ValidationError

from shortsmith import llm
from shortsmith.ledger import LedgerEntry

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
