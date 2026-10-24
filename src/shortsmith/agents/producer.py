"""The Producer: the director's plan -> every scene's image and voice (+ the approved hero clip) (6.2).

Plain code, not an agent. The Phase 3 agent producer spent 3-4 LLM turns (7-300 s each on the free
tier) deciding what code already knows: every scene needs an image and a voice. When the steps are
known, a workflow beats an agent: cheaper, faster, the same every time, and easy to resume.

It works like `make` (the build tool): media.json remembers which prompt made each image and which
words each voice file says. A file is (re)made only when it's missing or its input changed, so
- after a revision only the changed scenes are made again, and only the part that changed
  (new words -> new voice, same image; a new picture prompt -> new image, same voice);
- after a crash the node simply runs again: whatever was finished is skipped;
- the old file moves to history/round_N/ first, so before and after can be compared.

The tools are reached through call_tool(name, json_arguments): the same function the Phase 3 loop
uses. So the producer runs on direct registry calls, or through the MCP server (--mcp) unchanged:
media goes "via MCP" (PLAN §2) at the cost of one server start (~2 s) per producer pass.

Phase 7 adds two users of the same rules: `pending()` says what a pass WOULD make (the budget
estimates it before the node runs), and `only=` limits a pass to some scenes (`rerun-scene`).
"""

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

from shortsmith.agents import tools
from shortsmith.agents.director import HeroPlan
from shortsmith.media import ffmpeg, tts
from shortsmith.models import Scene, Script

CallTool = Callable[[str, str], str]  # (tool name, JSON arguments) -> short JSON result; raises on failure
MANIFEST = "media.json"


class Made(BaseModel):
    """What one producer pass made (scene numbers); everything else was already up to date."""

    images: list[int] = []
    voices: list[int] = []
    clips: list[int] = []

    def __str__(self) -> str:
        parts = [f"{kind} {', '.join(map(str, idx))}" for kind, idx in
                 (("images", self.images), ("voices", self.voices), ("clip", self.clips)) if idx]
        return "; ".join(parts) or "nothing (all up to date)"


def fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def load_manifest(run_dir: Path) -> dict[str, dict[str, str]]:
    path = run_dir / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(run_dir: Path, manifest: dict[str, dict[str, str]]) -> None:
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def history_folder(round_: int | str) -> str:
    """history/round_N/ for the graph's rounds; a label as it is (rerun-scene's "rerun_1")."""
    return f"round_{round_}" if isinstance(round_, int) else round_


def archive(run_dir: Path, relative: Path, round_: int | str) -> None:
    """Move an outdated file to history/round_N/ (N = the round that replaced it), if there is one."""
    source = run_dir / relative
    if source.exists():
        target = run_dir / "history" / history_folder(round_) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)


def clip_key(manifest_entry: dict[str, str], hero: HeroPlan) -> str:
    return fingerprint(manifest_entry.get("image", ""), hero.motion_prompt)


def clip_needed(run_dir: Path, plan: Script, hero: HeroPlan | None) -> bool:
    """Would producing this plan make a new (paid) clip? The approval step asks only then."""
    if hero is None or hero.approved is False:
        return False
    scene = next((s for s in plan.scenes if s.idx == hero.scene), None)
    if scene is None or scene.motion != "ai_clip":
        return False
    entry = dict(load_manifest(run_dir).get(str(hero.scene), {}))
    entry["image"] = fingerprint(scene.visual_prompt)  # the image this plan will have
    return entry.get("clip") != clip_key(entry, hero) or not (run_dir / ffmpeg.ai_clip_file(hero.scene)).exists()


def image_stale(run_dir: Path, entry: dict[str, str], scene: Scene) -> bool:
    return entry.get("image") != fingerprint(scene.visual_prompt) or not (run_dir / tools.image_file(scene.idx)).exists()


def voice_stale(run_dir: Path, entry: dict[str, str], scene: Scene, voice: str) -> bool:
    return entry.get("voice") != fingerprint(scene.narration, voice) or not (run_dir / tools.audio_file(scene.idx)).exists()


def pending(run_dir: Path, plan: Script, hero: HeroPlan | None, only: set[int] | None = None) -> Made:
    """What produce() would make right now, without making anything (the budget prices it first)."""
    manifest = load_manifest(run_dir)
    voice = tts.VOICES[plan.language]
    todo = Made()
    for scene in plan.scenes:
        if only is not None and scene.idx not in only:
            continue
        entry = manifest.get(str(scene.idx), {})
        if image_stale(run_dir, entry, scene):
            todo.images.append(scene.idx)
        if voice_stale(run_dir, entry, scene, voice):
            todo.voices.append(scene.idx)
    if hero and hero.approved and (only is None or hero.scene in only) and clip_needed(run_dir, plan, hero):
        todo.clips.append(hero.scene)
    return todo


def produce(run_dir: Path, plan: Script, hero: HeroPlan | None, call_tool: CallTool, round_: int | str = 0,
            only: set[int] | None = None) -> Made:
    """Make whatever is missing or out of date, scene by scene; the manifest is saved after every file.
    `only` limits the pass to some scenes (rerun-scene): the others are not even looked at."""
    (run_dir / "script.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")  # the tools read scenes here
    manifest = load_manifest(run_dir)
    voice = tts.VOICES[plan.language]
    made = Made()
    for scene in plan.scenes:
        if only is not None and scene.idx not in only:
            continue
        entry = manifest.setdefault(str(scene.idx), {})
        arguments = json.dumps({"scene": scene.idx})

        if image_stale(run_dir, entry, scene):
            archive(run_dir, tools.image_file(scene.idx), round_)  # generate_image reuses a file that exists
            call_tool("generate_image", arguments)
            entry["image"] = fingerprint(scene.visual_prompt)
            save_manifest(run_dir, manifest)
            made.images.append(scene.idx)

        if voice_stale(run_dir, entry, scene, voice):
            archive(run_dir, tools.audio_file(scene.idx), round_)
            call_tool("text_to_speech", arguments)
            entry["voice"] = fingerprint(scene.narration, voice)
            save_manifest(run_dir, manifest)
            made.voices.append(scene.idx)

    if hero and hero.approved and (only is None or hero.scene in only) and clip_needed(run_dir, plan, hero):
        entry = manifest[str(hero.scene)]
        archive(run_dir, ffmpeg.ai_clip_file(hero.scene), round_)
        call_tool("animate_image", json.dumps({"scene": hero.scene, "motion": hero.motion_prompt}))
        entry["clip"] = clip_key(entry, hero)
        save_manifest(run_dir, manifest)
        made.clips.append(hero.scene)
    return made


def adopt(run_dir: Path, plan: Script) -> list[int]:
    """Give a run made before the manifest existed (the straight pipeline, Phases 2-5) a media.json
    for the files it already has, so a rerun remakes only what is asked. Also writes the per-scene
    timing files the tools read (that pipeline saved one voiceovers.json). Returns the adopted scenes."""
    manifest = load_manifest(run_dir)
    voice = tts.VOICES[plan.language]
    saved = tts.load_voiceovers(run_dir) or []
    adopted = []
    for n, scene in enumerate(plan.scenes):
        entry = manifest.setdefault(str(scene.idx), {})
        before = dict(entry)
        if "image" not in entry and (run_dir / tools.image_file(scene.idx)).exists():
            entry["image"] = fingerprint(scene.visual_prompt)  # trusted: it was made from this script
        if "voice" not in entry and (run_dir / tools.audio_file(scene.idx)).exists():
            entry["voice"] = fingerprint(scene.narration, voice)
        timing = run_dir / tools.timing_file(scene.idx)
        if not timing.exists() and len(saved) == len(plan.scenes):
            relative = saved[n].model_copy(update={"audio_path": tools.audio_file(scene.idx)})
            timing.write_text(relative.model_dump_json(), encoding="utf-8")
        if entry != before:
            adopted.append(scene.idx)
    if adopted:
        save_manifest(run_dir, manifest)
    return adopted


@contextmanager
def tool_caller(run_dir: Path, via_mcp: bool = False) -> Iterator[CallTool]:
    """call_tool for one producer pass: direct registry calls, or the MCP server (one stdio subprocess)."""
    if via_mcp:
        from shortsmith.agents import mcp_tools  # the MCP SDK takes ~0.5 s to import: only when it's used

        with mcp_tools.McpToolbox.connect(mcp_tools.server_params()) as box:
            yield box.tool_caller(run_id=run_dir.name)
        return
    ctx: tools.ToolContext | None = None

    def call(name: str, arguments: str) -> str:
        nonlocal ctx
        if ctx is None:  # built on the first call, after produce() saved this round's script.json
            ctx = tools.ToolContext.for_run(run_dir)
        return tools.call_tool(ctx, name, arguments)

    yield call
