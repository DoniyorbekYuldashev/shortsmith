"""chat() with a fake OpenAI client: checks the ledger wiring without any API call or key."""

from types import SimpleNamespace as NS

import pytest

from shortsmith import llm
from shortsmith.ledger import LedgerEntry

MESSAGES = [{"role": "user", "content": "hi"}]


def _fake_client(response):
    """Mimics the one method chat() uses: client.chat.completions.create(**kwargs)."""
    return NS(chat=NS(completions=NS(create=lambda **kwargs: response)))


def test_chat_returns_usage_and_records_it(monkeypatch, tmp_path):
    response = NS(
        model="gpt-4.1-mini-2025-04-14",
        choices=[NS(message=NS(content="Hello!"))],
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
        choices=[NS(message=NS(content="Salom!"))],
        usage=NS(prompt_tokens=1_000_000, completion_tokens=0, prompt_tokens_details=None),
    )
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(response))

    result = llm.chat(MESSAGES, ledger_path=tmp_path / "ledger.jsonl")

    assert result.model == "gemini-3.5-flash-lite"
    assert result.cost_usd == pytest.approx(0.30)
