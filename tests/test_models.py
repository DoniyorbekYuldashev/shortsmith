"""The rules a Script must follow, checked without any LLM."""

import json

import pytest
from pydantic import ValidationError

from shortsmith.models import Script


def _scenes(words_each: int = 14, n: int = 5) -> list[dict]:
    narration = " ".join(["word"] * words_each)
    return [
        {"idx": i, "narration": narration, "visual_prompt": "...", "duration_s": 6, "motion": "zoom_in"}
        for i in range(1, n + 1)
    ]


def _script(**overrides) -> dict:
    """A valid 5-scene, 70-word (~30 s) English script; tests break one rule at a time."""
    data = {
        "title": "Autumn coffee in Tashkent",
        "hook": "Tashkent's coziest cup is hiding here",
        "scenes": _scenes(),
        "cta": "Follow for more Tashkent spots",
        "language": "en",
    }
    return data | overrides


def test_valid_script_passes():
    assert Script.model_validate(_script()).title == "Autumn coffee in Tashkent"


def test_long_hook_is_rejected():
    hook = "This is the most amazing coffee shop you will ever visit today"  # 12 words
    with pytest.raises(ValidationError, match="hook has 12 words"):
        Script.model_validate(_script(hook=hook))


def test_narration_must_be_about_30_seconds_of_words():
    with pytest.raises(ValidationError, match="150 words in total"):
        Script.model_validate(_script(scenes=_scenes(words_each=30)))  # ~60 s of speech


def test_word_targets_depend_on_the_language():
    scenes = _scenes(words_each=9)  # 45 words: too few for English, fine for Uzbek (longer words)
    with pytest.raises(ValidationError, match="45 words"):
        Script.model_validate(_script(scenes=scenes, language="en"))
    assert Script.model_validate(_script(scenes=scenes, language="uz")).language == "uz"


def test_planned_seconds_no_longer_decide_anything():
    scenes = [{**scene, "duration_s": 20} for scene in _scenes()]  # the LLM guesses 100 s: only a plan
    assert Script.model_validate(_script(scenes=scenes)).scenes[0].duration_s == 20


def test_saved_scripts_skip_the_length_rule_but_not_the_structure(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps(_script(scenes=_scenes(words_each=2))), encoding="utf-8")  # an older, shorter run
    assert len(Script.load(path).scenes) == 5
    path.write_text(json.dumps(_script(scenes=_scenes(n=2))), encoding="utf-8")
    with pytest.raises(ValidationError, match="2 scenes"):
        Script.load(path)


def test_too_few_scenes_are_rejected():
    with pytest.raises(ValidationError, match="2 scenes"):
        Script.model_validate(_script(scenes=_script()["scenes"][:2]))


def test_unknown_motion_is_rejected():
    scenes = _script()["scenes"]
    scenes[0] = {**scenes[0], "motion": "spin"}
    with pytest.raises(ValidationError):
        Script.model_validate(_script(scenes=scenes))


def test_extra_fields_are_rejected_and_the_schema_says_so():
    with pytest.raises(ValidationError):
        Script.model_validate(_script(mood="cozy"))
    assert Script.model_json_schema()["additionalProperties"] is False  # what the API is told
