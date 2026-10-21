"""The rules a Script must follow, checked without any LLM."""

import pytest
from pydantic import ValidationError

from shortsmith.models import Script


def _script(**overrides) -> dict:
    """A valid 5-scene, 30-second script; tests break one rule at a time."""
    data = {
        "title": "Autumn coffee in Tashkent",
        "hook": "Tashkent's coziest cup is hiding here",
        "scenes": [
            {"idx": i, "narration": "...", "visual_prompt": "...", "duration_s": 6, "motion": "zoom_in"}
            for i in range(1, 6)
        ],
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


def test_total_duration_must_fit_a_short():
    scenes = [{**scene, "duration_s": 20} for scene in _script()["scenes"]]  # 100 s
    with pytest.raises(ValidationError, match="in total"):
        Script.model_validate(_script(scenes=scenes))


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
