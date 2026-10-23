"""The spec for ledger.py: Step 1.3 is done when `uv run pytest` is all green."""

import pytest

from shortsmith import ledger
from shortsmith.ledger import LedgerEntry, cost_usd, record, total_cost


def test_cost_of_a_simple_call():
    # 1M in + 1M out on gpt-4.1-mini = $0.40 + $1.60
    assert cost_usd("gpt-4.1-mini", 1_000_000, 1_000_000) == pytest.approx(2.00)


def test_cached_tokens_are_part_of_the_prompt_and_cheaper():
    # 1M prompt tokens, 400k of them cached: 600k x $0.40 + 400k x $0.10 per 1M = $0.28
    assert cost_usd("gpt-4.1-mini", 1_000_000, 0, cached_tokens=400_000) == pytest.approx(0.28)


def test_dated_model_version_uses_the_base_price():
    assert cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)


def test_longest_matching_name_wins(monkeypatch):
    # "gpt-4.1-mini-..." starts with "gpt-4.1" too, but it must get the mini price
    prices = {**ledger.PRICES, "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00}}
    monkeypatch.setattr(ledger, "PRICES", prices)
    assert cost_usd("gpt-4.1-mini-2025-04-14", 1_000_000, 0) == pytest.approx(0.40)
    assert cost_usd("gpt-4.1-2025-04-14", 1_000_000, 0) == pytest.approx(2.00)


def test_unknown_model_raises_instead_of_costing_zero():
    with pytest.raises(ValueError):
        cost_usd("some-new-model", 100, 100)


def _entry(cost: float, step: str = "demo") -> LedgerEntry:
    return LedgerEntry(step=step, model="gpt-4.1-mini", prompt_tokens=10, completion_tokens=5, cost_usd=cost)


def test_record_appends_one_json_line_per_call(tmp_path):
    path = tmp_path / "runs" / "ledger.jsonl"  # the "runs" folder doesn't exist yet
    record(_entry(0.01), path)
    record(_entry(0.02, step="writer"), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert LedgerEntry.model_validate_json(lines[1]).step == "writer"


def test_total_cost_sums_the_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    for cost in (0.01, 0.02, 0.03):
        record(_entry(cost), path)
    assert total_cost(path) == pytest.approx(0.06)


def test_total_cost_of_missing_file_is_zero(tmp_path):
    assert total_cost(tmp_path / "nothing-yet.jsonl") == 0.0


def test_lines_written_inside_a_graph_run_carry_its_id(tmp_path):
    path = tmp_path / "ledger.jsonl"
    record(LedgerEntry(step="writer", model="m", prompt_tokens=1, completion_tokens=1, cost_usd=0.1), path)
    token = ledger.CURRENT_RUN.set("2026-10-23_221500")
    try:
        record(LedgerEntry(step="critic", model="m", prompt_tokens=1, completion_tokens=1, cost_usd=0.2), path)
    finally:
        ledger.CURRENT_RUN.reset(token)
    assert [e.step for e in ledger.entries(path, "2026-10-23_221500")] == ["critic"]
    assert [e.run_id for e in ledger.entries(path)] == [None, "2026-10-23_221500"]


def test_lines_from_before_phase_6_still_parse(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"ts": "2026-10-20T10:00:00Z", "step": "writer", "model": "m", "prompt_tokens": 1, '
                    '"cached_tokens": 0, "completion_tokens": 1, "cost_usd": 0.1}\n', encoding="utf-8")
    assert ledger.entries(path)[0].run_id is None and ledger.entries(tmp_path / "missing.jsonl") == []
