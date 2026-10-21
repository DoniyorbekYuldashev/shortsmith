"""Data passed between pipeline steps. The LLM fills these in; Pydantic checks them.

Two layers: the JSON schema sent to the API enforces the *shape* (fields, types, allowed
values), and the validators below enforce *rules* a schema can't express well. LLMs are
bad at counting words and seconds, so code checks those.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

Motion = Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "ai_clip"]

HOOK_MAX_WORDS = 10  # the prompt asks for 8: a little slack before rejecting
SCENES_MIN, SCENES_MAX = 4, 8
TOTAL_SECONDS = (20.0, 40.0)  # a ~30-second short


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")  # becomes "additionalProperties": false in the schema

    idx: int
    narration: str  # what the voice says
    visual_prompt: str  # prompt for the image model (always English)
    duration_s: float
    motion: Motion


class Script(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    hook: str  # the first ~2 seconds: the line that stops the scroll
    scenes: list[Scene]
    cta: str
    language: Literal["en", "uz", "ru"]

    @field_validator("hook")
    @classmethod
    def hook_is_short(cls, hook: str) -> str:
        words = len(hook.split())
        if words > HOOK_MAX_WORDS:
            raise ValueError(f"hook has {words} words, max {HOOK_MAX_WORDS} (it must fit ~2 seconds)")
        return hook

    @model_validator(mode="after")
    def scenes_fit_a_short(self) -> "Script":
        n = len(self.scenes)
        if not SCENES_MIN <= n <= SCENES_MAX:
            raise ValueError(f"{n} scenes, expected {SCENES_MIN}-{SCENES_MAX}")
        if [scene.idx for scene in self.scenes] != list(range(1, n + 1)):
            raise ValueError("scene idx must run 1, 2, 3, ... in order")
        total = sum(scene.duration_s for scene in self.scenes)
        low, high = TOTAL_SECONDS
        if not low <= total <= high:
            raise ValueError(f"scenes last {total:.0f} s in total, expected {low:.0f}-{high:.0f} s")
        return self
