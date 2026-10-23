"""The Reviser: the critic's reasons -> new words or a new picture prompt for the weak scenes only (6.4).

The optimizer half of the evaluator-optimizer loop. It has its own node between the Critic and
the Producer, rather than living inside either:
- not in the Critic: an evaluator that also rewrites tends to grade its own fix; the critic stays a judge;
- not in the Producer: the producer is plain code (no LLM), so re-running it after a crash is free;
- its own node = its own checkpoint: if the producer crashes after a revision, the revision
  isn't paid for again on resume, and the trace shows what the revision cost.

A regenerated scene must actually change: the critic's `fix` goes into the prompt, and code
compares the result with the old scene. An unchanged scene isn't regenerated (the producer only
remakes media whose prompt or narration changed).
"""

import re

from pydantic import BaseModel, ConfigDict

from shortsmith import llm
from shortsmith.agents.director import with_style
from shortsmith.llm import ChatResult, Message
from shortsmith.models import NARRATION_WORDS, SCENES_PLANNED, SceneIssue, Script

REVISER_SYSTEM_PROMPT = """You fix scenes of a ~30-second vertical short video that a critic flagged. You get \
the whole script for context and, for each flagged scene, what is wrong and a suggested fix. Rewrite ONLY \
the flagged scenes, and only what their problem says:
- problem "narration": new narration in the script's language, about the same number of words as before \
(the whole script must stay near its word target). Talk to the viewer; never describe the picture. Scene 1 \
must still start with a strong hook of at most 8 words; the last scene must still end with the call to action.
- problem "visual": a new visual_prompt in English: one vertical 9:16 image, concrete, no text in the image, \
ending with the same style words as the other scenes.
- problem "both": both.
hook: the first words of scene 1's narration, at most 8 words (the current hook if scene 1 isn't rewritten).
Copy unchanged fields exactly. A rewritten field must be clearly different from the old one and fix the reason given.
Never invent facts: no business names, addresses, prices, dates or numbers that are not already in the script. \
If a fix asks for one, make the line concrete another way (a detail, a tip, a reason), or drop the promise."""


class SceneFix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: int
    narration: str
    visual_prompt: str


class SceneFixes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook: str  # scene 1's opening words; the old hook when scene 1 isn't rewritten
    scenes: list[SceneFix]


class Revision(BaseModel):
    """One scene's change in one round: kept in the state (RunState.revisions, appended)."""

    round: int  # the critic round that asked for it
    scene: int
    problem: str
    reason: str
    fix: str
    narration_before: str
    narration_after: str
    visual_before: str
    visual_after: str

    @property
    def changed(self) -> bool:
        return self.narration_before != self.narration_after or self.visual_before != self.visual_after


def style_of(script: Script) -> str:
    """The style words all prompts end with: the longest common ending after the last comma-separated
    part that every prompt shares (the director appended them)."""
    tails = [[part.strip() for part in s.visual_prompt.split(",")] for s in script.scenes]
    common: list[str] = []
    for parts in zip(*(reversed(t) for t in tails)):
        if len(set(p.lower() for p in parts)) != 1:
            break
        common.insert(0, parts[0])
    return ", ".join(common)


def revise_messages(script: Script, issues: list[SceneIssue]) -> list[Message]:
    low, high = NARRATION_WORDS[script.language]
    lines = [f"Language: {llm.LANGUAGES[script.language]}; target {low}-{high} words in total "
             f"(~{round(low / SCENES_PLANNED)}-{round(high / SCENES_PLANNED)} per scene)",
             f"Hook: {script.hook}", f"Call to action: {script.cta}", ""]
    for scene in script.scenes:
        lines.append(f"Scene {scene.idx} ({len(scene.narration.split())} words): {scene.narration}")
        lines.append(f"  visual_prompt: {scene.visual_prompt}")
    lines += ["", "Flagged by the critic:"]
    lines += [f"- scene {i.scene}, problem {i.problem}: {i.reason} Suggested fix: {i.fix}" for i in issues]
    return [{"role": "system", "content": REVISER_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def _starts_with(text: str, start: str) -> bool:
    return bool(start.strip()) and text.lower().startswith(start.strip().rstrip(".!?").lower())


def apply_fixes(script: Script, issues: list[SceneIssue], fixes: SceneFixes, round_: int) -> tuple[Script, list[Revision]]:
    """The fixes on top of the script: only flagged scenes, only the fields their problem names."""
    wanted = {issue.scene: issue for issue in issues}
    given = {fix.scene: fix for fix in fixes.scenes if fix.scene in wanted}
    style = style_of(script)
    scenes, revisions = [], []
    hook, cta = script.hook, script.cta
    for scene in script.scenes:
        issue, fix = wanted.get(scene.idx), given.get(scene.idx)
        if not (issue and fix):
            scenes.append(scene)
            continue
        narration = (fix.narration.strip() or scene.narration) if issue.problem in ("narration", "both") else scene.narration
        visual = (with_style(fix.visual_prompt.strip(), style) if fix.visual_prompt.strip() else scene.visual_prompt)             if issue.problem in ("visual", "both") else scene.visual_prompt
        scenes.append(scene.model_copy(update={"narration": narration, "visual_prompt": visual}))
        revisions.append(Revision(round=round_, scene=scene.idx, problem=issue.problem, reason=issue.reason, fix=issue.fix,
                                  narration_before=scene.narration, narration_after=narration,
                                  visual_before=scene.visual_prompt, visual_after=visual))
        if scene.idx == script.scenes[0].idx and not _starts_with(narration, hook):
            # scene 1 starts with the hook, so a new opening is the new hook: the reviser's, if the narration
            # really starts with it, else the whole first sentence (never cut mid-phrase: run 1, E7)
            hook = fixes.hook.strip() if _starts_with(narration, fixes.hook) else _first_sentence(narration)
        if scene.idx == script.scenes[-1].idx and not narration.rstrip(" .!").endswith(cta.rstrip(" .!")):
            cta = re.split(r"(?<=[.!?])\s", narration.strip())[-1]
    revised = script.model_copy(update={"scenes": scenes, "hook": hook, "cta": cta})
    return revised, revisions


def revise(script: Script, issues: list[SceneIssue], round_: int,
           model: str | None = None) -> tuple[Script, list[Revision], ChatResult]:
    fixes, result = llm.paced(lambda: llm.chat_json(revise_messages(script, issues), SceneFixes, model=model, step="revise"))
    revised, revisions = apply_fixes(script, issues, fixes, round_)
    return revised, revisions, result
