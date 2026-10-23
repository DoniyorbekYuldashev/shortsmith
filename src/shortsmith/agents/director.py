"""The Director: the chosen script -> final shots (Step 6.2).

The Writer thinks in words; the Director thinks in pictures. With the words fixed, one call
decides what every scene looks like: a single style for all images (visual consistency, PLAN §7),
image prompts written for an image model (subject, composition, light; no text in the picture),
a camera move per still image, and which ONE scene becomes a paid AI clip (the "hero", usually
the hook). Code then applies the plan: the narration never changes here, every prompt ends with
the style words, and the hero scene's motion becomes "ai_clip".

Context engineering: the Director reads the script and the brand's style words only. No
research notes, no other options, no trends: none of that changes what a picture should show.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from shortsmith import llm
from shortsmith.llm import ChatResult, Message
from shortsmith.media import animate
from shortsmith.models import Script

KenBurns = Literal["zoom_in", "zoom_out", "pan_left", "pan_right"]  # never ai_clip: code marks the hero

DIRECTOR_SYSTEM_PROMPT = """You are the director of a ~30-second vertical short video. The script is final: \
you don't change a word of it. You decide what every scene LOOKS like.

For every scene:
- visual_prompt: one vertical 9:16 image for an image model, in English: subject, setting, composition \
(close-up, medium or wide shot; camera angle), light, lens. Concrete and specific to the topic, and \
consistent with the other scenes (the same place, people and time of day unless the story moves). No text, \
letters or logos in the image. It shows what the narration is about; it doesn't repeat the narration. The \
writer's picture idea is a rough note: rewrite it as a shot (vary close-ups and wide shots across scenes); \
never hand it back unchanged.
- motion: the camera move over the still image: zoom_in (focus on a detail), zoom_out (reveal a place), \
pan_left or pan_right (follow along a row of things or a street).

Also:
- style: the style words every visual_prompt ends with. If brand style words are given, use them word for word.
- hero_scene: the ONE scene that becomes a real 5-second AI video clip. It costs money, so only one: \
usually scene 1 (the hook), unless another scene has movement a still image can't show.
- hero_motion: what moves in that clip and how, one short English sentence about THIS scene's subject \
(the subject's movement, then the camera's).
- hero_reason: one sentence on why this scene."""


class Shot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idx: int
    visual_prompt: str
    motion: KenBurns


class DirectorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: str
    shots: list[Shot]
    hero_scene: int
    hero_motion: str
    hero_reason: str


class HeroPlan(BaseModel):
    """The one paid clip of a video, and whether a person said yes to it (Step 6.6)."""

    scene: int
    motion_prompt: str  # what moves, for the image-to-video model
    reason: str
    fallback: KenBurns  # the camera move used instead if the clip is refused
    cost_usd: float  # the estimate the approval question shows
    approved: bool | None = None  # None = not asked yet; False sticks for the rest of the run


def director_messages(script: Script, visual_style: str = "") -> list[Message]:
    lines = [f"Language of the narration: {llm.LANGUAGES[script.language]}", f"Title: {script.title}", ""]
    for scene in script.scenes:
        lines.append(f"Scene {scene.idx}: {scene.narration}")
        lines.append(f"  writer's picture idea: {scene.visual_prompt}")
    if visual_style.strip():
        lines += ["", f"Brand style words (use them word for word): {visual_style.strip()}"]
    return [{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


def with_style(prompt: str, style: str) -> str:
    """The prompt ending with the style words, adding only the ones it doesn't have yet (a director that
    already ended a prompt with "photorealistic, 8k" got them twice in run 3: E15)."""
    have = prompt.lower()
    missing = [part.strip() for part in style.split(",") if part.strip(" .") and part.strip(" .").lower() not in have]
    return f"{prompt.rstrip(' .,')}, {', '.join(missing)}" if missing else prompt


def apply_plan(script: Script, plan: DirectorPlan, visual_style: str = "") -> tuple[Script, HeroPlan]:
    """The director's shots on top of the writer's script. Narration is untouched; a scene the
    director skipped keeps the writer's prompt; a hero scene that doesn't exist becomes scene 1."""
    style = visual_style.strip() or plan.style.strip()
    shots = {shot.idx: shot for shot in plan.shots}
    indexes = [scene.idx for scene in script.scenes]
    hero_idx = plan.hero_scene if plan.hero_scene in indexes else indexes[0]
    scenes, fallback = [], "zoom_in"
    for scene in script.scenes:
        shot = shots.get(scene.idx)
        motion = shot.motion if shot else (scene.motion if scene.motion != "ai_clip" else "zoom_in")
        if scene.idx == hero_idx:
            fallback, motion = motion, "ai_clip"
        prompt = with_style(shot.visual_prompt if shot else scene.visual_prompt, style)
        scenes.append(scene.model_copy(update={"visual_prompt": prompt, "motion": motion}))
    hero = HeroPlan(scene=hero_idx, motion_prompt=plan.hero_motion.strip() or "slow natural movement",
                    reason=plan.hero_reason, fallback=fallback, cost_usd=animate.CLIP_PRICE_USD)
    return script.model_copy(update={"scenes": scenes}), hero


def direct(script: Script, visual_style: str = "",
           model: str | None = None) -> tuple[Script, HeroPlan, DirectorPlan, ChatResult]:
    """One call: the shot list for the chosen script (the raw plan is returned too, for director.json)."""
    plan, result = llm.paced(lambda: llm.chat_json(director_messages(script, visual_style), DirectorPlan,
                                                   model=model, step="director"))
    final, hero = apply_plan(script, plan, visual_style)
    return final, hero, plan, result
