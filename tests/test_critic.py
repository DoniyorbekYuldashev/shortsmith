"""The Critic (frames, request, code checks), the Director (applying a shot list) and the Reviser."""

import base64
from pathlib import Path

import pytest
from PIL import Image

from shortsmith.agents import critic, director, reviser, writer
from shortsmith.media import ffmpeg
from shortsmith.media.tts import Voiceover, Word
from shortsmith.models import CriticVerdict, HookIdea, ResearchNotes, SceneIssue, Script


def script(n: int = 6, words: int = 13, language: str = "en", style: str = "warm light, 35mm") -> Script:
    scenes = [{"idx": i, "narration": ("Stop scrolling now. " if i == 1 else "") + " ".join(["w"] * words),
               "visual_prompt": f"picture {i}, {style}", "duration_s": 5, "motion": "zoom_in"} for i in range(1, n + 1)]
    return Script.model_validate({"title": "T", "hook": "Stop scrolling now.", "scenes": scenes, "cta": "w",
                                  "language": language}, context={"saved": True})


def voiceovers(n: int = 6, last_word_end: float = 4.0) -> list[Voiceover]:
    return [Voiceover(audio_path=Path(f"{i}.mp3"), duration_s=last_word_end + 0.5,
                      words=[Word(text="w", start_s=0.1, end_s=last_word_end)]) for i in range(n)]


# --- which frames ----------------------------------------------------------------------------------


@pytest.mark.parametrize("indexes, focus, expected", [
    ([1, 2, 3, 4, 5, 6], [], [1, 3, 4, 6]),  # spread: always the hook and the last scene
    ([1, 2, 3, 4, 5, 6], [2], [1, 2, 3, 4]),  # a regenerated scene first, then the spread
    ([1, 2, 3, 4, 5, 6], [5, 2, 6], [1, 2, 5, 6]),
    ([1, 2, 3], [], [1, 2, 3]),  # fewer scenes than frames: all of them
    ([1, 2, 3, 4, 5, 6], [9], [1, 3, 4, 6]),  # a scene that doesn't exist is ignored
])
def test_pick_scenes(indexes, focus, expected):
    assert critic.pick_scenes(indexes, focus) == expected


def test_frames_are_cut_in_the_middle_of_each_scene_on_the_editors_timeline():
    times = critic.frame_times(voiceovers(3, last_word_end=4.0), [1, 2, 3])
    # each scene = last word 4.0 s + 0.3 s pause = 129 frames (4.3 s); the last one holds 0.7 s longer
    assert times == {1: 2.15, 2: 6.45, 3: 11.1}


def test_the_frame_command_seeks_then_scales_to_half_width():
    cmd = critic.frame_cmd(Path("final.mp4"), 6.45, Path("critic/round_0/scene_02.jpg"))
    assert cmd[cmd.index("-ss") + 1] == "6.450" and cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-vf") + 1] == "scale=540:-2" and cmd[cmd.index("-frames:v") + 1] == "1"
    assert cmd[-1] == "critic/round_0/scene_02.jpg"


def test_sample_frames_cuts_real_frames_from_a_video(tmp_path):
    """The one critic test that runs ffmpeg: a 4-scene, 18-second test video."""
    ffmpeg.run(ffmpeg.FFMPEG + ["-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30:duration=18", "-pix_fmt", "yuv420p",
                                "final.mp4"], cwd=tmp_path)
    frames = critic.sample_frames(tmp_path, voiceovers(4), script(4), round_=0)
    assert [(idx, t) for idx, t, _ in frames] == [(1, 2.15), (2, 6.45), (3, 10.75), (4, 15.4)]
    with Image.open(frames[0][2]) as image:
        assert image.size == (540, 960) and image.format == "JPEG"


# --- the request ------------------------------------------------------------------------------------


def test_the_vision_request_has_text_then_labelled_images(tmp_path):
    jpeg = tmp_path / "f.jpg"
    Image.new("RGB", (4, 4), "blue").save(jpeg)
    s = script()
    messages = critic.critic_messages("coffee", s, critic.checks(s), [(1, 2.2, jpeg), (3, 9.1, jpeg)], mock=True)

    assert messages[0] == {"role": "system", "content": critic.CRITIC_SYSTEM_PROMPT}
    parts = messages[1]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text", "image_url"]
    assert "PLACEHOLDER CARDS" in parts[0]["text"] and "Measured by code: words per scene" in parts[0]["text"]
    assert parts[1]["text"] == "Frame from scene 1 (2.2 s into the video):"
    url = parts[2]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,") and base64.b64decode(url.split(",", 1)[1]) == jpeg.read_bytes()


def test_without_frames_it_reviews_the_script_alone():
    s = script()
    [_, user] = critic.critic_messages("coffee", s, critic.checks(s), [], mock=True)
    assert len(user["content"]) == 1 and critic.SCRIPT_NOTE in user["content"][0]["text"]
    assert "PLACEHOLDER" not in user["content"][0]["text"]


def test_code_checks_words_and_copied_hooks(monkeypatch):
    monkeypatch.setattr(writer, "_library", {})
    notes = ResearchNotes(trending=[], hooks=[HookIdea(pattern="Stop", hook="Stop scrolling now!", why="w")],
                          angles=[], avoid=[], visual_style="")
    c = critic.checks(script(words=9), notes)
    assert c.words == [12, 9, 9, 9, 9, 9] and c.total_words == 57 and c.target == (78, 84) and not c.in_range
    assert c.hook_copied_from == "notes: Stop"
    assert "total 57 (target 78-84: too short)" in critic.facts_text(c)


def test_the_verdict_is_clamped_and_cut_not_rejected():
    issues = [{"scene": s, "problem": "visual", "reason": "r", "fix": "f"} for s in (2, 2, 3, 4, 5)]
    v = CriticVerdict.model_validate({"score": 12, "hook": "h", "summary": "s", "cannot_judge": "c", "weak_scenes": issues})
    assert v.score == 10 and [i.scene for i in v.weak_scenes] == [2, 3, 4]  # one per scene, the worst 3


def test_the_critic_schema_is_a_list_of_issues_not_a_dict():
    schema = CriticVerdict.model_json_schema()
    assert schema["properties"]["weak_scenes"]["type"] == "array"
    assert "additionalProperties" not in schema["properties"]["weak_scenes"]


# --- the director ---------------------------------------------------------------------------------------


def shots(n: int = 6, hero: int = 1, skip: int | None = None) -> director.DirectorPlan:
    return director.DirectorPlan(style="cinematic", hero_scene=hero, hero_motion="steam rises", hero_reason="hook",
                                 shots=[director.Shot(idx=i, visual_prompt=f"shot {i}", motion="pan_right")
                                        for i in range(1, n + 1) if i != skip])


def test_the_director_changes_pictures_never_words():
    s = script()
    final, hero = director.apply_plan(s, shots(), visual_style="warm amber light")
    assert [sc.narration for sc in final.scenes] == [sc.narration for sc in s.scenes]
    assert final.scenes[1].visual_prompt == "shot 2, warm amber light"  # the brand's words win over the director's
    assert final.scenes[0].motion == "ai_clip" and hero.fallback == "pan_right" and hero.scene == 1
    assert hero.cost_usd == 0.30 and hero.approved is None


def test_a_skipped_shot_keeps_the_writers_prompt_and_a_bad_hero_becomes_scene_1():
    final, hero = director.apply_plan(script(), shots(hero=9, skip=4))
    assert final.scenes[3].visual_prompt == "picture 4, warm light, 35mm, cinematic" and hero.scene == 1


def test_style_words_are_added_once():
    assert director.with_style("a cup, cinematic", "cinematic") == "a cup, cinematic"
    assert director.with_style("a cup.", "cinematic") == "a cup, cinematic"
    assert director.with_style("a cup", "") == "a cup"


# --- the reviser -------------------------------------------------------------------------------------------


def issue(scene: int, problem: str) -> SceneIssue:
    return SceneIssue(scene=scene, problem=problem, reason="describes the picture", fix="give a tip instead")


def test_only_flagged_scenes_and_only_the_named_field_change():
    s = script()
    fixes = reviser.SceneFixes(hook="Stop scrolling now.", scenes=[
        reviser.SceneFix(scene=2, narration="New words for two", visual_prompt="should be ignored"),
        reviser.SceneFix(scene=3, narration="ignored", visual_prompt="a new picture"),
        reviser.SceneFix(scene=5, narration="not flagged", visual_prompt="not flagged"),
    ])
    revised, revisions = reviser.apply_fixes(s, [issue(2, "narration"), issue(3, "visual")], fixes, round_=0)

    assert revised.scenes[1].narration == "New words for two" and revised.scenes[1].visual_prompt == s.scenes[1].visual_prompt
    assert revised.scenes[2].narration == s.scenes[2].narration
    assert revised.scenes[2].visual_prompt == "a new picture, warm light, 35mm"  # the shared style words stay
    assert revised.scenes[4] == s.scenes[4]
    assert [(r.scene, r.changed) for r in revisions] == [(2, True), (3, True)]


def test_an_unchanged_rewrite_is_marked_as_unchanged():
    s = script()
    fixes = reviser.SceneFixes(hook="Stop scrolling now.", scenes=[reviser.SceneFix(scene=2, narration=s.scenes[1].narration, visual_prompt="x")])
    _, [revision] = reviser.apply_fixes(s, [issue(2, "narration")], fixes, round_=1)
    assert revision.changed is False and revision.round == 1


def test_a_new_opening_becomes_the_new_hook():
    s = script()
    fixes = reviser.SceneFixes(hook="Stop scrolling now.", scenes=[reviser.SceneFix(scene=1, narration="Nobody tells you this. Then more words.",
                                                        visual_prompt="x")])
    revised, _ = reviser.apply_fixes(s, [issue(1, "narration")], fixes, round_=0)
    assert revised.hook == "Nobody tells you this."  # the reviser kept the old hook field: the first sentence wins
    fixes = fixes.model_copy(update={"hook": "Nobody tells you"})
    assert reviser.apply_fixes(s, [issue(1, "narration")], fixes, round_=0)[0].hook == "Nobody tells you"
    long = reviser.SceneFixes(hook="x", scenes=[reviser.SceneFix(
        scene=1, narration="Step into the coziest autumn vibe with this rich spiced pumpkin latte.", visual_prompt="x")])
    assert reviser.apply_fixes(s, [issue(1, "narration")], long, round_=0)[0].hook.endswith("pumpkin latte.")  # not cut


def test_the_style_is_the_shared_ending_of_every_prompt():
    assert reviser.style_of(script(style="warm amber light, terracotta, 35mm film look")) == \
        "warm amber light, terracotta, 35mm film look"


def test_the_reviser_prompt_lists_the_issues_and_word_counts():
    [_, user] = reviser.revise_messages(script(), [issue(3, "both")])
    assert "Scene 3 (13 words):" in user["content"]
    assert "- scene 3, problem both: describes the picture Suggested fix: give a tip instead" in user["content"]


def test_only_the_missing_style_words_are_added():
    assert director.with_style("a desk, photorealistic, 8k resolution", "photorealistic, 8k resolution, cinematic") == \
        "a desk, photorealistic, 8k resolution, cinematic"


def test_no_developer_docstring_reaches_the_models_schema():
    from shortsmith.agents import writer as w
    for schema in (CriticVerdict, w.ScriptOptions, w.WriterPick, director.DirectorPlan, reviser.SceneFixes):
        assert "description" not in str(schema.model_json_schema())
