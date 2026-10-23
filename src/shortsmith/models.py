"""Data passed between pipeline steps. The LLM fills these in; Pydantic checks them.

Two layers: the JSON schema sent to the API enforces the *shape* (fields, types, allowed
values), and the validators below enforce *rules* a schema can't express well. LLMs are
bad at counting words and seconds, so code checks those.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

Motion = Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "ai_clip"]

HOOK_MAX_WORDS = 10  # the prompt asks for 8: a little slack before rejecting
SCENES_MIN, SCENES_MAX = 4, 8

# Narration words (all scenes together) for ~30 s on screen. Speech length is set by the
# word count, not by the LLM's duration_s guesses, which ran ~40% long in Step 2.2. Measured
# on edge-tts in Step 2.4, pauses included: ~2.7 words/s in English, ~2.0 in Uzbek (longer
# words). Russian is an estimate until a Russian run is measured.
NARRATION_WORDS = {"en": (78, 84), "uz": (55, 60), "ru": (58, 64)}
WORDS_SLACK = 0.25  # reject only when far off: the real length is measured from the voice anyway
SCENES_PLANNED = 6  # the prompt asks for 6 (= 6 images per video); 4-8 still pass


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")  # becomes "additionalProperties": false in the schema

    idx: int
    narration: str  # what the voice says
    visual_prompt: str  # prompt for the image model (always English)
    duration_s: float  # the LLM's plan; the video uses the real voice length
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
    def scenes_fit_a_short(self, info: ValidationInfo) -> "Script":
        n = len(self.scenes)
        if not SCENES_MIN <= n <= SCENES_MAX:
            raise ValueError(f"{n} scenes, expected {SCENES_MIN}-{SCENES_MAX}")
        if [scene.idx for scene in self.scenes] != list(range(1, n + 1)):
            raise ValueError("scene idx must run 1, 2, 3, ... in order")
        if (info.context or {}).get("saved"):
            return self  # an older run's script: the length targets may have changed since it was written
        words = sum(len(scene.narration.split()) for scene in self.scenes)
        low, high = NARRATION_WORDS[self.language]
        if not low * (1 - WORDS_SLACK) <= words <= high * (1 + WORDS_SLACK):
            raise ValueError(f"narration has {words} words in total, expected about {low}-{high} ({self.language}, ~30 s)")
        return self

    @classmethod
    def load(cls, path: Path) -> "Script":
        """Read a script.json saved by a run. Structure is re-checked; the word-count rule is
        for fresh LLM output only, so old runs can still be re-assembled."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"), context={"saved": True})


# --- research notes (Phase 5): what the Researcher hands the Writer ------------------------------

# The notes go into the writer's prompt, so they are kept short: a model asked for "2-4" items
# sometimes writes 6, and extras are cut rather than rejected (the call is paid either way).
NOTES_MAX = {"trending": 4, "hooks": 3, "angles": 3, "avoid": 4}


class HookIdea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str  # a retrieved pattern's name, exactly as in the knowledge base
    hook: str  # that pattern applied to this topic, in the video's language
    why: str  # one sentence: why it fits this topic and audience


class ResearchNotes(BaseModel):
    """Compact, evidence-based notes: real trends + retrieved hook patterns (+ brand rules)."""

    model_config = ConfigDict(extra="forbid")

    trending: list[str]  # what gets views for this topic now, from the real top videos
    hooks: list[HookIdea]  # the patterns that fit best, best first
    angles: list[str]  # concrete angles or details the script can use
    avoid: list[str]  # clichés, saturated angles, the brand's words to avoid
    visual_style: str  # the brand's style words for every image prompt; "" without a brand

    @field_validator("trending", "hooks", "angles", "avoid")
    @classmethod
    def keep_it_short(cls, items: list, info: ValidationInfo) -> list:
        return items[: NOTES_MAX[info.field_name]]

    def for_writer(self) -> str:
        """The notes as the few lines the writer reads (context engineering: no scores, no ids, no URLs)."""
        lines = ["Research notes (real YouTube trends of the last 30 days, and proven hook patterns):", "What gets views now:"]
        lines += [f"- {item}" for item in self.trending]
        lines.append("Hook patterns that fit. Base your hook on one of them (adapt it, don't copy it word for word):")
        lines += [f"- {h.pattern}: \"{h.hook}\" ({h.why})" for h in self.hooks]
        lines.append("Angles to use:")
        lines += [f"- {item}" for item in self.angles]
        lines.append("Avoid:")
        lines += [f"- {item}" for item in self.avoid]
        if self.visual_style.strip():
            lines.append(f"End every visual_prompt with these style words: {self.visual_style.strip()}")
        return "\n".join(lines)
