"""rerun.plan_change and runs.resolve: what a rerun will do, worked out before anything is made."""

import json

import pytest

from shortsmith import rerun, runs
from shortsmith.agents.director import HeroPlan
from shortsmith.models import Script

WORDS = "one two three four five six seven eight nine ten eleven twelve thirteen"


def script() -> Script:
    scenes = [{"idx": i, "narration": f"Stop scrolling now. {WORDS}" if i == 1 else f"{WORDS}. Visit today.",
               "visual_prompt": f"shot {i}, cinematic, warm light", "duration_s": 5,
               "motion": "ai_clip" if i == 1 else "zoom_in"} for i in range(1, 7)]
    return Script.model_validate({"title": "t", "hook": "Stop scrolling now.", "scenes": scenes, "cta": "Visit today.",
                                  "language": "en"})


HERO = HeroPlan(scene=1, motion_prompt="steam rises", reason="the hook", fallback="pan_left", cost_usd=0.30, approved=True)


def test_no_option_redoes_the_image_and_the_voice_as_they_are():
    plan = rerun.plan_change(script(), rerun.RerunRequest(scene=3))
    assert (plan.image, plan.voice) == (True, True) and plan.script == script()


def test_a_new_prompt_keeps_the_runs_style_words_and_redoes_only_the_image():
    plan = rerun.plan_change(script(), rerun.RerunRequest(scene=3, prompt="close-up of hands pouring tea"))
    assert plan.script.scenes[2].visual_prompt == "close-up of hands pouring tea, cinematic, warm light"
    assert (plan.image, plan.voice) == (True, False) and plan.hero_question is None


def test_the_same_text_again_is_nothing_to_redo_unless_forced():
    same = rerun.RerunRequest(scene=2, narration=script().scenes[1].narration)
    with pytest.raises(ValueError, match="nothing to redo"):
        rerun.plan_change(script(), same)
    assert rerun.plan_change(script(), same.model_copy(update={"voice": True})).voice


def test_new_words_for_scene_1_make_a_new_hook():
    plan = rerun.plan_change(script(), rerun.RerunRequest(scene=1, narration=f"Coffee costs less here. {WORDS}"))
    assert plan.script.hook == "Coffee costs less here."
    too_long = rerun.RerunRequest(scene=1, narration="This opening sentence is far too long to ever be a hook. Yes.")
    with pytest.raises(ValueError, match="hook"):
        rerun.plan_change(script(), too_long)


def test_new_words_for_the_last_scene_make_a_new_call_to_action():
    plan = rerun.plan_change(script(), rerun.RerunRequest(scene=6, narration=f"{WORDS}. Come by tonight."))
    assert plan.script.cta == "Come by tonight."


def test_a_new_picture_for_the_approved_hero_scene_asks_about_the_clip():
    plan = rerun.plan_change(script(), rerun.RerunRequest(scene=1, prompt="a new hook picture"), HERO)
    assert plan.hero_question and "no longer matches" in plan.hero_question["question"]
    assert rerun.plan_change(script(), rerun.RerunRequest(scene=1, voice=True), HERO).hero_question is None  # same picture
    refused = HERO.model_copy(update={"approved": False})
    assert rerun.plan_change(script(), rerun.RerunRequest(scene=1, image=True), refused).hero_question is None


def test_a_missing_scene_is_a_clear_error():
    with pytest.raises(ValueError, match="no scene 7: this run has scenes 1-6"):
        rerun.plan_change(script(), rerun.RerunRequest(scene=7))


def test_resolve_finds_runs_by_id_prefix_latest_or_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("2026-10-23_100000", "2026-10-24_100000", "2026-10-24_110000"):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / "script.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "runs" / "experiments").mkdir()  # not a run: no run files
    assert runs.resolve("latest").name == "2026-10-24_110000"
    assert runs.resolve("2026-10-23").name == "2026-10-23_100000"
    assert runs.resolve("runs/2026-10-24_100000").name == "2026-10-24_100000"
    with pytest.raises(runs.RunNotFound, match="matches 2 runs"):
        runs.resolve("2026-10-24")
    with pytest.raises(runs.RunNotFound, match="no run 'experiments'"):
        runs.resolve("experiments")
