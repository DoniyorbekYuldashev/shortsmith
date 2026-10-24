"""budget.py and costs.py: estimates before a node runs, the stop rule, and the ledger grouped for `cost`.
Hand-made ledger lines and states; no LLM, no graph run."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from shortsmith import budget, costs, ledger
from shortsmith.agents.director import HeroPlan
from shortsmith.agents.reviser import Revision
from shortsmith.agents.state import RunState
from shortsmith.models import CriticReport, ScriptChecks, Script

NOW = datetime(2026, 10, 24, 15, 0, tzinfo=timezone.utc)  # 08:00 in California (PDT): the quota day began 07:00 UTC


def line(step: str, usd: float, model: str = "gemini-3.5-flash-lite", run_id: str | None = None,
         ts: datetime = NOW, **tokens: int) -> ledger.LedgerEntry:
    return ledger.LedgerEntry(ts=ts, step=step, model=model, prompt_tokens=tokens.get("p", 0),
                              completion_tokens=tokens.get("c", 0), cost_usd=usd, run_id=run_id)


@pytest.fixture(autouse=True)
def gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MOCK_MEDIA", "1")
    monkeypatch.delenv("GEMINI_TIER", raising=False)
    monkeypatch.setattr(budget.critic, "CRITIC_MODEL", "gemini-3.8-flash")


def script() -> Script:
    scenes = [{"idx": i, "narration": "one two three four five six seven eight nine ten eleven twelve thirteen",
               "visual_prompt": f"shot {i}, warm light", "duration_s": 5, "motion": "zoom_in"} for i in range(1, 7)]
    return Script.model_validate({"title": "t", "hook": "one two three", "scenes": scenes, "cta": "thirteen",
                                  "language": "en"})


def state(tmp_path, **fields) -> RunState:
    return RunState(run_id="r1", run_dir=str(tmp_path), topic="coffee", **fields)


# --- estimates ---------------------------------------------------------------------------------------------


def test_an_estimate_is_the_most_expensive_of_the_last_10_calls():
    history = [line("writer", 0.009), line("writer", 0.008)] + [line("writer", 0.001 * n) for n in range(1, 11)]
    usd, known = budget.call_cost("writer", "gemini-3.5-flash-lite", history)
    assert known and usd == pytest.approx(0.010)  # the two old expensive calls fell out of the window


def test_only_the_same_step_on_the_same_model_counts_and_dated_names_match():
    history = [line("writer", 0.5, model="gemini-3.8-flash"), line("director", 0.5),
               line("writer", 0.002, model="gpt-4.1-mini-2025-04-14")]
    assert budget.call_cost("writer", "gpt-4.1-mini", history) == (0.002, True)


def test_without_history_the_estimate_is_a_first_call_guess():
    usd, known = budget.call_cost("director", "gemini-3.5-flash-lite", [])
    assert not known and usd == ledger.cost_usd("gemini-3.5-flash-lite", *budget.FIRST_CALL_TOKENS["director"])


def test_the_writer_node_counts_its_options_call_and_its_pick():
    est = budget.llm_estimate("writer", [line("writer", 0.004), line("writer-pick", 0.0006)])
    assert est.usd == pytest.approx(0.0046)
    assert est.basis == "writer $0.0040 + writer-pick $0.0006 (most expensive of the last 10)"


def test_the_critic_is_priced_on_the_critic_model():
    history = [line("critic", 0.001), line("critic", 0.007, model="gemini-3.8-flash")]
    assert budget.llm_estimate("critic", history).usd == 0.007


def test_nodes_without_llm_calls_are_free():
    for name in ("approve_hero", "editor"):
        assert budget.llm_estimate(name, []).usd == 0


def test_the_producer_is_free_with_mock_media_and_priced_with_real_media(tmp_path, monkeypatch):
    s = state(tmp_path, plan=script(), hero=HeroPlan(scene=1, motion_prompt="m", reason="r", fallback="zoom_in",
                                                    cost_usd=0.30, approved=True))
    assert budget.estimate("producer", s, []).usd == 0  # placeholders: 6 images, 6 voices, all free
    monkeypatch.setenv("MOCK_MEDIA", "0")
    est = budget.estimate("producer", s, [])
    assert est.usd == pytest.approx(6 * 0.003)  # the hero scene is still Ken Burns in this plan: no clip
    assert "images 1, 2, 3, 4, 5, 6" in est.basis and "real media" in est.basis


# --- the stop rule -------------------------------------------------------------------------------------------


def test_check_stops_when_spent_plus_the_estimate_passes_the_budget(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    for entry in (line("critic", 0.01, model="gemini-3.8-flash"), line("writer", 0.003, run_id="r1"),
                  line("director", 0.002, run_id="r1"), line("writer", 0.9, run_id="other run")):
        ledger.record(entry, path)
    s = state(tmp_path, budget_usd=0.012)
    with pytest.raises(budget.BudgetExceeded) as stop:
        budget.check("critic", s, path)
    assert (stop.value.node, stop.value.spent, stop.value.estimate, stop.value.limit) == ("critic", 0.005, 0.01, 0.012)
    assert str(stop.value) == ("over budget before critic: spent $0.0050 + critic ~$0.0100 > budget $0.0120 "
                               "(list price)")
    assert budget.check("critic", state(tmp_path, budget_usd=0.015), path).usd == 0.01  # fits: no stop


def test_a_free_node_is_never_stopped_and_no_budget_never_stops(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.record(line("writer", 1.0, run_id="r1"), path)  # already far over
    assert budget.check("editor", state(tmp_path, budget_usd=0.01), path).usd == 0
    assert budget.check("writer", state(tmp_path), path).usd > 0  # budget_usd=None


def test_what_is_left_of_a_run(tmp_path):
    fresh = state(tmp_path, research_on=True)
    assert budget.nodes_left(fresh) == ["researcher", "writer", "director", "critic", "critic", "critic", "revise", "revise"]
    report = CriticReport(score=5, hook="h", weak_scenes=[], summary="s", cannot_judge="c", round=0, mode="video",
                          frames=[1], model="m", checks=ScriptChecks(words=[1], total_words=1, target=(1, 2), in_range=False,
                                                                     hook_words=1, hook_copied_from=None))
    revision = Revision(round=0, scene=3, problem="visual", reason="r", fix="f", narration_before="a", narration_after="a",
                        visual_before="b", visual_after="c")
    later = state(tmp_path, chosen=script(), plan=script(), critic_rounds=[report], revisions=[revision])
    assert budget.nodes_left(later) == ["critic", "critic", "revise"]


def test_a_whole_run_estimate_for_the_real_question():
    est = budget.run_estimate(research=False, hero_clips=1, max_rounds=2, real=True, history=[])
    assert (est.images, est.images_usd, est.clip_usd) == (6, pytest.approx(0.018), 0.30)
    assert est.revision_images == 6 and est.most_usd > est.first_pass_usd > 0.318
    mock = budget.run_estimate(research=False, hero_clips=1, max_rounds=2, real=False, history=[])
    assert mock.images_usd == mock.clip_usd == mock.revision_usd == 0 and mock.llm_usd > 0


# --- costs ----------------------------------------------------------------------------------------------------------


def ledger_lines() -> list[ledger.LedgerEntry]:
    yesterday = NOW - timedelta(days=1)
    return [
        line("writer", 0.004, run_id="2026-10-24_100000", p=800, c=2000),
        line("critic", 0.005, model="gemini-3.8-flash", run_id="2026-10-24_100000", p=4500, c=200),
        line("critic", 0.005, model="gemini-3.8-flash", ts=NOW - timedelta(hours=9), run_id="2026-10-24_000000"),  # before 07:00 UTC
        line("writer", 0.002, model="gpt-4.1-mini-2025-04-14", ts=yesterday),
        line("agent", 0.003, model="fal-ai/flux/schnell", ts=yesterday, run_id="2026-10-23_120000"),
    ]


def test_billed_is_zero_for_gemini_on_the_free_tier_and_the_list_price_for_the_rest(monkeypatch):
    gemini, openai_, fal = ledger_lines()[0], ledger_lines()[3], ledger_lines()[4]
    assert (costs.billed_usd(gemini), costs.billed_usd(openai_), costs.billed_usd(fal)) == (0, 0.002, 0.003)
    monkeypatch.setenv("GEMINI_TIER", "paid")
    assert costs.billed_usd(gemini) == 0.004


def test_the_report_groups_by_day_run_and_model_step():
    r = costs.report(ledger_lines(), now=NOW, quota_log=Path("no-such-quota.jsonl"))
    assert (r.total.calls, r.total.list_usd, r.total.billed_usd) == (5, pytest.approx(0.019), pytest.approx(0.005))
    assert [row.key for row in r.by_run] == ["2026-10-24_100000", "2026-10-24_000000", "2026-10-23_120000", "(no run id)"]
    assert r.by_model_step[0].key == "gemini-3.8-flash | critic" and r.by_model_step[0].calls == 2
    assert r.by_day[0].key >= r.by_day[-1].key  # newest first
    assert r.total.completion_tokens == 2200


def test_days_and_run_filters():
    assert costs.report(ledger_lines(), days=1, now=NOW, quota_log=Path("no-such-quota.jsonl")).total.calls <= 3
    one = costs.report(ledger_lines(), run="2026-10-24_100000", now=NOW, quota_log=Path("no-such-quota.jsonl"))
    assert one.total.calls == 2 and one.run == "2026-10-24_100000"


def test_quotas_count_only_since_midnight_pacific(tmp_path):
    q = costs.report(ledger_lines(), now=NOW, critic_model="gemini-3.6-flash", quota_log=tmp_path / "none.jsonl").quotas
    assert q.day_start == datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)
    today = {m.model: (m.requests, m.daily_limit) for m in q.gemini}
    assert today["gemini-3.8-flash"] == (1, 20)  # the 06:00 UTC call belongs to yesterday's quota day
    assert today["gemini-3.6-flash"] == (0, 20) and today["gemini-3.5-flash-lite"] == (1, None)
    assert q.youtube_units == 0 and q.youtube_cap == 2000
    assert any("can't be asked" in note for note in q.notes)


def test_status_of_a_snapshot():
    from shortsmith import runs

    def snap(next_=(), interrupts=(), errors=(), values=True):
        return NS(values={"run_id": "r"} if values else {}, next=next_, interrupts=list(interrupts),
                  tasks=[NS(error=e) for e in errors])

    assert runs.status_of(None).label == "no checkpoint"
    assert runs.status_of(snap(values=False)).label == "no checkpoint"
    assert runs.status_of(snap()).label == "finished"
    paused = runs.status_of(snap(("approve_hero",), [NS(value={"question": "Scene 1 ...?"})]))
    assert (paused.label, paused.detail) == ("paused at approval", "Scene 1 ...?")
    over = runs.status_of(snap(("critic",), errors=['"BudgetExceeded(\'over budget before critic: spent $0.0050\')"']))
    assert (over.label, over.detail) == ("over budget before critic", "over budget before critic: spent $0.0050")
    crashed = runs.status_of(snap(("critic",), errors=['"DailyQuotaError(\'used up\')"']))
    assert crashed.label == "crashed at critic" and "DailyQuotaError" in crashed.detail
    killed = runs.status_of(snap(("producer",)))
    assert killed.label == "crashed at producer" and "without an error" in killed.detail
