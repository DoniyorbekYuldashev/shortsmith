"""The agent's tools: plain Python functions, each with a Pydantic model for its arguments.

The model never runs anything. It only asks for a call (a tool name plus JSON arguments);
`call_tool()` checks the arguments against the tool's model, runs the function and returns a
short JSON result. The JSON schema the API sees is generated from that same model, so the
schema and the validation can't disagree (the same idea as chat_json's response_format).

Tools work inside one run folder (ToolContext) and take scene numbers, never file paths:
the model can't touch anything outside the run, and its arguments stay a few tokens long.
Every result goes back into the conversation and is paid for again on every later turn, so
results are short JSON (`{"ok": true, "path": "scenes/scene_02.png", ...}`), never logs.

Phase 4 moves these tools into an MCP server: the registry already holds what MCP needs
for each one (name, description, input schema, function).
"""

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from shortsmith.llm import Message
from shortsmith.media import ffmpeg, images, tts
from shortsmith.models import Scene, Script


class ToolContext(BaseModel):
    """What every tool call shares: one run folder and its script. The model never sees this."""

    run_dir: Path
    script: Script
    voiceovers: dict[int, tts.Voiceover] = {}  # scene idx -> its voice, filled by text_to_speech
    fail_first: set[str] = set()  # demo switch (--fail-first): these tools fail on their first call
    failed: set[str] = set()  # ...and these already did

    @classmethod
    def for_run(cls, run_dir: Path, fail_first: set[str] | None = None) -> "ToolContext":
        """Bind a run folder that has a script.json; a complete saved voice is picked up too."""
        script = Script.load(run_dir / "script.json")
        saved = tts.load_voiceovers(run_dir) or []
        voiceovers = {}
        if len(saved) == len(script.scenes):  # an older script's voice doesn't count
            voiceovers = {scene.idx: v for scene, v in zip(script.scenes, saved)}
        return cls(run_dir=run_dir, script=script, voiceovers=voiceovers, fail_first=fail_first or set())

    def scene(self, idx: int) -> Scene:
        for scene in self.script.scenes:
            if scene.idx == idx:
                return scene
        raise ValueError(f"no scene {idx}: this script has scenes 1-{len(self.script.scenes)}")


def image_file(idx: int) -> Path:
    """Where a scene's image lives, relative to the run folder (ffmpeg.assemble reads it there)."""
    return Path("scenes") / f"scene_{idx:02d}.png"


def audio_file(idx: int) -> Path:
    return Path("audio") / f"scene_{idx:02d}.mp3"


class Tool(BaseModel):
    name: str
    description: str  # written for the model: it decides when to call the tool from this text
    args: type[BaseModel]
    fn: Callable[[ToolContext, Any], BaseModel]


TOOLS: dict[str, Tool] = {}


def tool[A: BaseModel](
    args: type[A],
) -> Callable[[Callable[[ToolContext, A], BaseModel]], Callable[[ToolContext, A], BaseModel]]:
    """Register a function as a tool. Its docstring becomes the description the model reads
    (FastMCP's @mcp.tool() in Phase 4 does the same)."""

    def register(fn: Callable[[ToolContext, A], BaseModel]) -> Callable[[ToolContext, A], BaseModel]:
        TOOLS[fn.__name__] = Tool(name=fn.__name__, description=inspect.getdoc(fn) or "", args=args, fn=fn)
        return fn

    return register


# --- arguments: what the model may send ---------------------------------------------------


class SceneArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown argument is an error the model gets to read

    scene: int = Field(ge=1, description="Scene number (idx) from the script, starting at 1.")


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- results: what goes back to the model (short!) ----------------------------------------


class ImageResult(BaseModel):
    scene: int
    path: str  # relative to the run folder: short, and the model never needs a full path
    reused: bool  # True = the image already existed; a paid image is never made twice


class VoiceResult(BaseModel):
    scene: int
    path: str
    duration_s: float
    words: int


class VideoResult(BaseModel):
    path: str
    duration_s: float
    size_mb: float
    scenes: int


# --- the tools ---------------------------------------------------------------------------


@tool(SceneArgs)
def generate_image(ctx: ToolContext, args: SceneArgs) -> ImageResult:
    """Make the vertical 9:16 image for one scene, from the script's visual prompt.
    Saves scenes/scene_NN.png. An image that already exists is reused, not made again."""
    scene = ctx.scene(args.scene)
    path = image_file(scene.idx)
    reused = (ctx.run_dir / path).exists()
    if not reused:
        images.generate_image(scene.visual_prompt, ctx.run_dir / path, f"Scene {scene.idx} - {scene.motion}", step="agent")
    return ImageResult(scene=scene.idx, path=path.as_posix(), reused=reused)


@tool(SceneArgs)
def text_to_speech(ctx: ToolContext, args: SceneArgs) -> VoiceResult:
    """Record the voiceover for one scene, from the script's narration.
    Saves audio/scene_NN.mp3 plus its word timings (used for the subtitles)."""
    scene = ctx.scene(args.scene)
    path = audio_file(scene.idx)
    voiceover = tts.synthesize(scene.narration, tts.VOICES[ctx.script.language], ctx.run_dir / path)
    ctx.voiceovers[scene.idx] = voiceover
    if len(ctx.voiceovers) == len(ctx.script.scenes):  # complete: `make --run` can now re-edit without voicing again
        tts.save_voiceovers([ctx.voiceovers[s.idx] for s in ctx.script.scenes], ctx.run_dir)
    return VoiceResult(
        scene=scene.idx, path=path.as_posix(), duration_s=round(voiceover.duration_s, 2), words=len(voiceover.words)
    )


@tool(NoArgs)
def render_video(ctx: ToolContext, args: NoArgs) -> VideoResult:
    """Edit everything into final.mp4: camera motion, voice, music and subtitles.
    Every scene needs its image and its voiceover first."""
    scenes = ctx.script.scenes
    no_image = [s.idx for s in scenes if not (ctx.run_dir / image_file(s.idx)).exists()]
    no_voice = [s.idx for s in scenes if s.idx not in ctx.voiceovers]
    missing = []  # a precise error lets the model fix exactly what's missing
    if no_image:
        missing.append(f"no image for scenes {no_image}")
    if no_voice:
        missing.append(f"no voiceover for scenes {no_voice}")
    if missing:
        raise FileNotFoundError(f"not ready to render: {'; '.join(missing)}")
    final = ffmpeg.assemble(ctx.run_dir, ctx.script, [ctx.voiceovers[s.idx] for s in scenes])
    info = ffmpeg.probe(final)
    return VideoResult(
        path=final.name, duration_s=round(info.duration_s, 1), size_mb=round(info.size_bytes / 1e6, 1), scenes=len(scenes)
    )


# --- what the API sees, and how a call is run -----------------------------------------------


def _without_titles(node: Any) -> Any:
    """Drop Pydantic's auto "title" keys ("Scene", "SceneArgs"): they tell the model nothing,
    and the schemas are re-sent (and paid for) on every turn. Property names are kept."""
    if isinstance(node, dict):
        return {
            key: {name: _without_titles(s) for name, s in value.items()} if key in ("properties", "$defs")
            else _without_titles(value)
            for key, value in node.items()
            if key != "title"
        }
    if isinstance(node, list):
        return [_without_titles(item) for item in node]
    return node


def schema(t: Tool) -> Message:
    """One tool in the Chat Completions "function" format (Gemini's compatible endpoint takes it too)."""
    return {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": _without_titles(t.args.model_json_schema())},
    }


def schemas() -> list[Message]:
    return [schema(t) for t in TOOLS.values()]


def call_tool(ctx: ToolContext, name: str, arguments: str) -> str:
    """Run one tool call the model asked for; return its result as short JSON.

    Raises on an unknown name, invalid arguments or a failing tool. The loop catches that
    and sends the error back to the model as the result, so the model can retry or adapt.
    """
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name!r}; the tools are: {', '.join(TOOLS)}")
    t = TOOLS[name]
    args = t.args.model_validate_json(arguments or "{}")  # the model's JSON is input like any other: validate it
    if name in ctx.fail_first and name not in ctx.failed:
        ctx.failed.add(name)
        raise ConnectionError("the service closed the connection without an answer")  # injected (--fail-first)
    result = t.fn(ctx, args)
    return json.dumps({"ok": True, **result.model_dump(mode="json")}, ensure_ascii=False)
