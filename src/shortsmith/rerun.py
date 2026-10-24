"""rerun-scene (Step 7.1): redo one scene of a run and cut the video again, paying for that scene only.

    shortsmith rerun-scene <run> 3 --prompt "a wide shot of the bazaar at dawn"

It goes through the producer's manifest (media.json), like a graph round does: the new prompt or
narration goes into script.json, the producer is limited to that one scene (only={3}), whatever it
replaces moves to history/rerun_N/ (the graph uses history/round_N/), and the editor cuts final.mp4
again (the old video goes to the same folder). The other scenes are never looked at, so nothing
else is made or paid for, even if it were out of date.

No LLM call: the new words come from the person. The checkpoint is left alone (a finished graph run
stays finished, and its state keeps the plan the graph left); reruns.json records every rerun, and
`shortsmith show` reads it. If the scene is the hero scene and its picture changes, the old clip no
longer matches it. A new clip is a paid step, so it is asked like the graph asks: yes = a new clip,
no = the director's Ken Burns move for that scene.
"""

import re
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from shortsmith import ledger
from shortsmith.agents import producer, reviser, tools
from shortsmith.agents.director import HeroPlan, with_style
from shortsmith.media import ffmpeg, images, tts
from shortsmith.models import Scene, Script

RERUNS = "reruns.json"


class RerunRequest(BaseModel):
    """What the person asked for. No text and no flag = redo the image and the voice as they are."""

    scene: int
    prompt: str | None = None  # a new picture prompt (the run's style words are added if missing)
    narration: str | None = None  # new spoken words
    image: bool = False  # remake the image even if its prompt is unchanged (a real model draws a new one)
    voice: bool = False


class RerunPlan(BaseModel):
    """What a rerun will do, worked out before anything is made (the --real question shows it)."""

    script: Script  # the new script.json
    scene: int
    image: bool
    voice: bool
    hero_question: dict[str, Any] | None = None  # set when the hero scene's picture changes: a new paid clip


class RerunRecord(BaseModel):
    """One rerun, as reruns.json keeps it."""

    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    label: str  # history/<label>/ holds the files it replaced
    scene: int
    made: str  # what the producer made, e.g. "images 3; voices 3"
    prompt_before: str
    prompt_after: str
    narration_before: str
    narration_after: str
    motion_before: str
    motion_after: str
    cost_usd: float  # list price of the ledger lines it added (0 with mock media)


class NeedsApproval(RuntimeError):
    """The rerun would make a new paid hero clip and nobody answered: nothing was done."""

    def __init__(self, question: dict[str, Any]):
        self.question = question
        super().__init__(question["question"])


_RECORDS = TypeAdapter(list[RerunRecord])


def load_records(run_dir: Path) -> list[RerunRecord]:
    path = run_dir / RERUNS
    return _RECORDS.validate_json(path.read_bytes()) if path.exists() else []


def find(script: Script, idx: int) -> Scene:
    for scene in script.scenes:
        if scene.idx == idx:
            return scene
    raise ValueError(f"no scene {idx}: this run has scenes 1-{len(script.scenes)}")


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def plan_change(script: Script, req: RerunRequest, hero: HeroPlan | None = None) -> RerunPlan:
    """The new script and what has to be remade. Raises ValueError when nothing would change."""
    old = find(script, req.scene)
    prompt = with_style(req.prompt.strip(), reviser.style_of(script)) if req.prompt and req.prompt.strip() else old.visual_prompt
    narration = req.narration.strip() if req.narration and req.narration.strip() else old.narration
    asked = bool(req.prompt or req.narration or req.image or req.voice)
    redo_image = req.image or prompt != old.visual_prompt or not asked
    redo_voice = req.voice or narration != old.narration or not asked
    if not (redo_image or redo_voice):
        raise ValueError("nothing to redo: the new text is the same as the old one "
                         "(add --image or --voice to make that part again anyway)")

    hook, cta = script.hook, script.cta
    if old.idx == script.scenes[0].idx and narration != old.narration and not narration.lower().startswith(
            hook.strip().rstrip(".!?").lower()):
        hook = _first_sentence(narration)  # scene 1 starts with the hook: a new opening is the new hook
    if old.idx == script.scenes[-1].idx and narration != old.narration and not narration.rstrip(" .!").endswith(
            cta.rstrip(" .!")):
        cta = re.split(r"(?<=[.!?])\s", narration)[-1]  # the last scene ends with the call to action
    scenes = [s.model_copy(update={"visual_prompt": prompt, "narration": narration}) if s.idx == old.idx else s
              for s in script.scenes]
    try:  # model_copy doesn't validate: check the hook rule (and the structure) before anything is made
        new = Script.model_validate({**script.model_dump(), "scenes": [s.model_dump() for s in scenes],
                                     "hook": hook, "cta": cta}, context={"saved": True})
    except ValidationError as e:
        problems = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors(include_url=False))
        raise ValueError(f"the new scene {old.idx} breaks a script rule: {problems} "
                         f"(scene 1's first sentence is the hook)") from e

    question = None
    if hero and hero.approved and hero.scene == old.idx and old.motion == "ai_clip" and redo_image:
        mock = images.mock_media()
        price = f"~${hero.cost_usd:.2f}" + (" with real media; MOCK_MEDIA=1 makes a free placeholder clip now" if mock else "")
        question = {"question": f"Scene {old.idx} is the hero scene and its picture changes, so its 5-second AI clip "
                                f"({hero.motion_prompt}) no longer matches. A new clip costs {price}. Make it?",
                    "scene": old.idx, "motion": hero.motion_prompt, "cost_usd": hero.cost_usd, "mock": mock}
    return RerunPlan(script=new, scene=old.idx, image=redo_image, voice=redo_voice, hero_question=question)


def next_label(run_dir: Path) -> str:
    n = 1
    while (run_dir / "history" / f"rerun_{n}").exists():
        n += 1
    return f"rerun_{n}"


def rerun_scene(run_dir: Path, req: RerunRequest, hero: HeroPlan | None = None,
                approve: Callable[[dict[str, Any]], Any] | None = None, via_mcp: bool = False,
                ledger_path: Path = ledger.DEFAULT_PATH) -> RerunRecord:
    """Remake one scene's image and/or voice, then the video. Raises NeedsApproval (nothing done) when
    a new hero clip would be needed and `approve` gives no answer."""
    script = Script.load(run_dir / "script.json")
    plan = plan_change(script, req, hero)
    new, idx = plan.script, plan.scene
    old_scene = find(script, idx)
    if plan.hero_question:
        answer = approve(plan.hero_question) if approve else None
        if answer is None:
            raise NeedsApproval(plan.hero_question)
        from shortsmith.agents.graph import is_yes  # the graph's own reading of an answer ("ha" = yes in Uzbek)

        if not is_yes(answer):
            hero = hero.model_copy(update={"approved": False})
            new = new.model_copy(update={"scenes": [s.model_copy(update={"motion": hero.fallback}) if s.idx == idx else s
                                                    for s in new.scenes]})

    producer.adopt(run_dir, script)  # a run from before media.json: its files count as made from script.json
    label = next_label(run_dir)
    if plan.image and find(new, idx).visual_prompt == old_scene.visual_prompt:
        producer.archive(run_dir, tools.image_file(idx), label)  # same prompt, made again: the file has to go first
    if plan.voice and find(new, idx).narration == old_scene.narration:
        producer.archive(run_dir, tools.audio_file(idx), label)
    if find(new, idx).motion != "ai_clip":
        producer.archive(run_dir, ffmpeg.ai_clip_file(idx), label)  # a refused clip no longer belongs to the scene

    lines_before = len(ledger.entries(ledger_path, run_dir.name))
    token = ledger.CURRENT_RUN.set(run_dir.name)  # a paid image is charged to this run, like in the graph
    try:
        with producer.tool_caller(run_dir, via_mcp) as call_tool:
            made = producer.produce(run_dir, new, hero, call_tool, label, only={idx})
        voiceovers = tts.load_voiceovers(run_dir)
        if voiceovers is None or len(voiceovers) != len(new.scenes):
            raise FileNotFoundError(f"{run_dir}: not every scene has its voice, so the video can't be cut again")
        final = run_dir / "final.mp4"
        if final.exists():
            (run_dir / "history" / label).mkdir(parents=True, exist_ok=True)
            shutil.copy(final, run_dir / "history" / label / "final.mp4")
        ffmpeg.assemble(run_dir, new, voiceovers)
    finally:
        ledger.CURRENT_RUN.reset(token)

    added = ledger.entries(ledger_path, run_dir.name)[lines_before:]
    after = find(new, idx)
    record = RerunRecord(label=label, scene=idx, made=str(made),
                         prompt_before=old_scene.visual_prompt, prompt_after=after.visual_prompt,
                         narration_before=old_scene.narration, narration_after=after.narration,
                         motion_before=old_scene.motion, motion_after=after.motion,
                         cost_usd=sum(e.cost_usd for e in added))
    records = load_records(run_dir) + [record]
    (run_dir / RERUNS).write_bytes(_RECORDS.dump_json(records, indent=2))
    return record
