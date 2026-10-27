"""budget.py and costs.py: estimates before a node runs, the stop rule, and the ledger grouped for `cost`.
Hand-made ledger lines and states; no LLM, no graph run."""

from datetime import date, datetime, timedelta, timezone
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
    for name in ("IMAGE_PROVIDER", "OPENAI_IMAGE_MODEL", "OPENAI_IMAGE_QUALITY", "OPENAI_IMAGE_SIZE", "FAL_IMAGE_MODEL",
                 *costs.ALLOWANCE_SETTINGS):
        monkeypatch.delenv(name, raising=False)
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
    assert est.usd == pytest.approx(6 * 0.005)  # gpt-image-2, low; the hero scene is still Ken Burns here: no clip
    assert "images 1, 2, 3, 4, 5, 6" in est.basis and "real media: images openai gpt-image-2 (low, 1024x1536)" in est.basis
    monkeypatch.setenv("IMAGE_PROVIDER", "fal")
    monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")
    assert budget.estimate("producer", s, []).usd == pytest.approx(6 * 0.003)


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


def test_a_whole_run_estimate_for_the_real_question(monkeypatch):
    monkeypatch.setenv("IMAGE_PROVIDER", "fal")
    monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")
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


# --- OpenAI's free daily tokens (data sharing) ---------------------------------------------------------------------


def sharing(monkeypatch, mini: int = 1000, large: int = 500, action: str = "warn") -> None:
    monkeypatch.setenv("OPENAI_DATA_SHARING", "1")
    monkeypatch.setenv("OPENAI_FREE_DAILY_TOKENS_MINI", str(mini))
    monkeypatch.setenv("OPENAI_FREE_DAILY_TOKENS_LARGE", str(large))
    monkeypatch.setenv("OPENAI_FREE_LIMIT_ACTION", action)


def test_openai_text_is_free_within_the_days_allowance_and_billed_beyond_it(monkeypatch):
    sharing(monkeypatch)  # 1,000 mini tokens a day
    first = line("writer", 0.004, model="gpt-5-mini-2025-08-07", ts=NOW, p=400, c=200)  # 600: free
    second = line("director", 0.006, model="gpt-5-mini-2025-08-07", ts=NOW + timedelta(minutes=1), p=500, c=300)  # 800: 400 free
    third = line("revise", 0.002, model="gpt-5-mini", ts=NOW + timedelta(minutes=2), p=100, c=100)  # all beyond
    embed = line("rag", 0.0001, model="text-embedding-3-small", ts=NOW, p=50)  # never free
    billing = costs.Billing([third, embed, second, first])  # any order: billing walks them by time
    assert billing(first) == 0
    assert billing(second) == pytest.approx(0.006 * 400 / 800)  # the share of its tokens past the line
    assert (billing(third), billing(embed)) == (0.002, 0.0001)


def test_the_allowance_is_per_group_and_refreshes_at_midnight_utc(monkeypatch):
    sharing(monkeypatch, mini=1000, large=500)
    late = datetime(2026, 10, 26, 23, 59, tzinfo=timezone.utc)
    lines = [line("writer", 0.01, model="gpt-5-mini", ts=late, p=1000),  # uses up the mini group
             line("critic", 0.02, model="gpt-5.1-2025-11-13", ts=late, p=500),  # the large group is separate
             line("writer", 0.01, model="gpt-4.1-mini", ts=late + timedelta(minutes=2), p=1000)]  # 00:01 UTC: a new day
    billing = costs.Billing(lines)
    assert [billing(e) for e in lines] == [0, 0, 0]
    assert costs.allowance_day(late + timedelta(minutes=2)) == date(2026, 10, 27)
    tashkent = timezone(timedelta(hours=5))  # 04:59 in Tashkent is still the 26th's allowance in UTC
    assert costs.allowance_day(datetime(2026, 10, 27, 4, 59, tzinfo=tashkent)) == date(2026, 10, 26)


def test_without_data_sharing_every_openai_call_is_billed_at_list_price(monkeypatch):
    lines = [line("writer", 0.01, model="gpt-5-mini", p=100)]
    assert costs.Billing(lines)(lines[0]) == 0.01 and costs.allowance().data_sharing is False
    assert "every call is billed at list price" in costs.billing_note() or costs.billing_note().startswith("Gemini")


def test_models_outside_the_groups_are_never_free(monkeypatch):
    sharing(monkeypatch)
    for model in ("gpt-5.5", "gpt-image-2", "text-embedding-3-small", "gemini-3.8-flash"):
        assert costs.free_group(model) is None
    assert (costs.free_group("gpt-5-mini-2025-08-07"), costs.free_group("gpt-5.1")) == ("mini", "large")


def test_the_allowance_view_counts_today_per_group(monkeypatch):
    sharing(monkeypatch, mini=1000, large=500)
    lines = [line("writer", 0.004, model="gpt-5-mini", p=700, c=500),  # 1,200 > 1,000: 200 of them billed
             line("critic", 0.01, model="gpt-5.1", p=300),
             line("writer", 0.004, model="gpt-5-mini", ts=NOW - timedelta(days=1), p=900)]  # yesterday
    a = costs.report(lines, now=NOW, quota_log=Path("no-such-quota.jsonl")).quotas.openai
    groups = {g.group: g for g in a.groups}
    assert a.data_sharing and a.day_start == datetime(2026, 10, 24, tzinfo=timezone.utc)
    assert (groups["mini"].tokens, groups["mini"].left, groups["mini"].models) == (1200, 0, ["gpt-5-mini"])
    assert groups["mini"].billed_usd == pytest.approx(0.004 * 200 / 1200)
    assert (groups["large"].tokens, groups["large"].left, groups["large"].billed_usd) == (300, 200, 0)
    assert any("can't be asked" in note for note in a.notes)


def test_the_report_bills_openai_within_the_allowance_as_zero(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    sharing(monkeypatch, mini=10_000)
    lines = [line("writer", 0.004, model="gpt-5-mini", run_id="r", p=800, c=2000)]
    r = costs.report(lines, now=NOW, quota_log=Path("no-such-quota.jsonl"))
    assert (r.total.list_usd, r.total.billed_usd) == (0.004, 0) and "data sharing" in r.billing


def test_a_node_that_would_pass_the_allowance_warns_or_stops(monkeypatch, caplog):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(budget.critic, "CRITIC_MODEL", None)
    sharing(monkeypatch, mini=3000, large=10_000)
    history = [line("writer", 0.001, model="gpt-5-mini", p=2000, c=500, ts=NOW)]  # 2,500 of 3,000 used today
    assert budget.allowance_check("director", history, now=NOW) is not None  # ~1,250 more (first-call guess)
    assert "past today's free OpenAI tokens" in caplog.text
    assert budget.allowance_check("critic", history, now=NOW) is None  # the large group has room for ~6,000
    assert budget.allowance_check("editor", history, now=NOW) is None  # no LLM call
    sharing(monkeypatch, mini=3000, large=10_000, action="stop")
    with pytest.raises(budget.FreeAllowanceReached, match="mini group: 2,500 used of 3,000"):
        budget.allowance_check("director", history, now=NOW)
    monkeypatch.setenv("OPENAI_DATA_SHARING", "0")
    assert budget.allowance_check("director", history, now=NOW) is None  # off: list price is billed anyway


def test_a_whole_run_estimate_with_openai_images(monkeypatch):
    est = budget.run_estimate(research=False, hero_clips=0, max_rounds=1, real=True, history=[])
    assert est.images_usd == pytest.approx(6 * 0.005) and est.revision_usd == pytest.approx(3 * 0.005)
    assert est.image_source.startswith("openai gpt-image-2 (low, 1024x1536)")
    monkeypatch.setenv("OPENAI_IMAGE_QUALITY", "medium")
    assert budget.run_estimate(False, 0, 0, True, []).images_usd == pytest.approx(6 * 0.041)
