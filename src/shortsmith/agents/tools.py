"""The agent's tools: plain Python functions, each with a Pydantic model for its arguments.

The model never runs anything. It only asks for a call (a tool name plus JSON arguments);
`call_tool()` checks the arguments against the tool's model, runs the function and returns a
short JSON result. The JSON schema the API sees is generated from that same model, so the
schema and the validation can't disagree (the same idea as chat_json's response_format).

Tools work inside one run folder (ToolContext) and take scene numbers, never file paths:
the model can't touch anything outside the run, and its arguments stay a few tokens long.
Every result goes back into the conversation and is paid for again on every later turn, so
results are short JSON (`{"ok": true, "path": "scenes/scene_02.png", ...}`), never logs.

Two front-ends use this one registry: the agent loop (Phase 3, direct calls) and the MCP
server (Phase 4, mcp_server.py), which publishes the same tools with a run_id added.

Tools come in two groups. "media" tools work inside one run folder (they get a ToolContext).
"research" tools (Phase 5: search_trends, retrieve_hooks) read the world and the knowledge
base, not a run, so they take only their arguments. The producer agent is given only the
media tools: research is the Researcher's job, and every schema it doesn't need costs tokens.
"""

import inspect
import json
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shortsmith.llm import Message
from shortsmith.media import ffmpeg, images, tts
from shortsmith.models import Scene, Script
from shortsmith.research import rag, youtube

Group = Literal["media", "research"]


class ToolContext(BaseModel):
    """What every tool call shares: one run folder and its script. The model never sees this."""

    run_dir: Path
    script: Script
    voiceovers: dict[int, tts.Voiceover] = {}  # scene idx -> its voice, filled by text_to_speech
    fail_first: set[str] = set()  # demo switch (--fail-first): these tools fail on their first call
    failed: set[str] = set()  # ...and these already did

    @classmethod
    def for_run(cls, run_dir: Path, fail_first: set[str] | None = None) -> "ToolContext":
        """Bind a run folder that has a script.json. Voice already made is picked up: the whole
        voiceovers.json `make` writes, or else each scene that text_to_speech saved on its own."""
        script = Script.load(run_dir / "script.json")
        saved = tts.load_voiceovers(run_dir) or []
        if len(saved) == len(script.scenes):  # an older script's voice doesn't count
            voiceovers = {scene.idx: v for scene, v in zip(script.scenes, saved)}
        else:
            voiceovers = {s.idx: v for s in script.scenes if (v := load_voiceover(run_dir, s.idx))}
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


def timing_file(idx: int) -> Path:
    """A scene's word timings, saved next to its audio. render_video needs them, and the MCP
    server loads a fresh ToolContext from disk for every call, so they can't live in memory only."""
    return Path("audio") / f"scene_{idx:02d}.json"


def load_voiceover(run_dir: Path, idx: int) -> tts.Voiceover | None:
    """One scene's saved voice, or None if its timings or its audio file are missing."""
    timing = run_dir / timing_file(idx)
    if not (timing.exists() and (run_dir / audio_file(idx)).exists()):
        return None
    saved = tts.Voiceover.model_validate_json(timing.read_text(encoding="utf-8"))
    return saved.model_copy(update={"audio_path": run_dir / saved.audio_path})


class Tool(BaseModel):
    name: str
    description: str  # written for the model: it decides when to call the tool from this text
    args: type[BaseModel]
    fn: Callable[..., BaseModel]  # media: fn(ctx, args); research: fn(args)
    group: Group = "media"


TOOLS: dict[str, Tool] = {}


def tool[F: Callable[..., BaseModel]](args: type[BaseModel], group: Group = "media") -> Callable[[F], F]:
    """Register a function as a tool. Its docstring becomes the description the model reads
    (FastMCP's @mcp.tool() in Phase 4 does the same)."""

    def register(fn: F) -> F:
        TOOLS[fn.__name__] = Tool(name=fn.__name__, description=inspect.getdoc(fn) or "", args=args, fn=fn, group=group)
        return fn

    return register


def names(group: Group) -> list[str]:
    return [t.name for t in TOOLS.values() if t.group == group]


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
    relative = voiceover.model_copy(update={"audio_path": path})  # like voiceovers.json: a moved run still works
    (ctx.run_dir / timing_file(scene.idx)).write_text(relative.model_dump_json(), encoding="utf-8")
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


# --- research tools (Phase 5): no run folder -------------------------------------------------


class TrendArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=80, description=(
        "What viewers would type into YouTube search: 2-5 keywords, e.g. 'Samarqand Registoni' or 'coffee shop Tashkent'."))
    language: Literal["en", "uz", "ru"] = Field("en", description="The video's language: sets the search's language and region hints.")


class TrendsResult(BaseModel):
    query: str
    since: date
    from_cache: bool
    units_used: int  # YouTube API quota spent by this call
    videos: list[youtube.TrendVideo]


class HookArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=300, description="The video's topic, or the kind of hook wanted. Any language.")
    k: int = Field(rag.DEFAULT_K, ge=1, le=10, description="How many hook patterns to return.")
    brand: str | None = Field(None, description=(
        "Also return this brand's most relevant guideline sections, e.g. 'anor-va-dolchin' (a café) or "
        "'samoviy-karvon' (a travel company). Leave it out for hook patterns only."))


class HooksResult(BaseModel):
    hooks: list[rag.Retrieved]
    brand_rules: list[rag.Retrieved] = []


@tool(TrendArgs, group="research")
def search_trends(args: TrendArgs) -> TrendsResult:
    """Find the most-viewed YouTube videos of Shorts length (3 minutes or less) from the last 30 days
    for a search: title, channel, views, length, date and link, most viewed first. Real data, cached
    for 24 hours. It is a keyword search: skip results about something else (a word can mean another
    thing in another language). A live search uses 1% of the day's API quota, so search a few times, not dozens."""
    report = youtube.search_trends(args.query, args.language)
    return TrendsResult(query=report.query, since=report.published_after.date(), from_cache=report.from_cache,
                        units_used=report.units_used, videos=report.videos)


@tool(HookArgs, group="research")
def retrieve_hooks(args: HookArgs) -> HooksResult:
    """Find the hook patterns (the first 2 seconds of a short video) that fit a topic best, from
    Shortsmith's library of ~70: name, template, examples in English, Uzbek and Russian, when to use it
    and why it works, with a similarity score (0-1, higher is closer). With a brand, also that brand's
    most relevant guideline sections (tone, words to use and avoid, visual style)."""
    rules = rag.brand_rules(args.query, args.brand) if args.brand else []  # first: an unknown brand fails before any hook is embedded
    return HooksResult(hooks=rag.retrieve(args.query, args.k), brand_rules=rules)


class RunStatus(BaseModel):
    """What a run folder already has: the agent's task message and the MCP status resource say it."""

    images: list[int]  # scene numbers
    voiceovers: list[int]
    final: bool  # final.mp4 exists


def run_status(ctx: ToolContext) -> RunStatus:
    return RunStatus(
        images=[s.idx for s in ctx.script.scenes if (ctx.run_dir / image_file(s.idx)).exists()],
        voiceovers=sorted(ctx.voiceovers),
        final=(ctx.run_dir / "final.mp4").exists(),
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


def schemas(group: Group = "media") -> list[Message]:
    """One group's tools for the API. The default is what the producer agent gets: the media tools."""
    return [schema(t) for t in TOOLS.values() if t.group == group]


def run_tool(ctx: ToolContext | None, name: str, arguments: str) -> BaseModel:
    """Run one tool call the model asked for and return its result model.

    Raises on an unknown name, invalid arguments or a failing tool. The loop catches that
    and sends the error back to the model as the result, so the model can retry or adapt;
    the MCP server sends it back as an error result (isError) instead. Research tools need
    no run (ctx may be None); media tools do.
    """
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name!r}; the tools are: {', '.join(TOOLS)}")
    t = TOOLS[name]
    args = t.args.model_validate_json(arguments or "{}")  # the model's JSON is input like any other: validate it
    if t.group == "research":
        return t.fn(args)
    if ctx is None:
        raise ValueError(f"{name} works inside a run, and no run was given")
    if name in ctx.fail_first and name not in ctx.failed:
        ctx.failed.add(name)
        raise ConnectionError("the service closed the connection without an answer")  # injected (--fail-first)
    return t.fn(ctx, args)


def ok_json(result: dict[str, Any]) -> str:
    """A successful result as the short JSON a Chat Completions tool message carries.
    That message is only a string, so success has to be said inside it ("ok")."""
    return json.dumps({"ok": True, **result}, ensure_ascii=False)


def call_tool(ctx: ToolContext | None, name: str, arguments: str) -> str:
    """run_tool() for the agent loop: the result as short JSON. Raises like run_tool()."""
    return ok_json(run_tool(ctx, name, arguments).model_dump(mode="json"))


# --- failures, as the model reads them ------------------------------------------------------

MAX_ERROR_CHARS = 300  # an error is read by the model and paid for on every later turn


class ToolFailed(Exception):
    """A failure already described for the model elsewhere (by an MCP server): describe_error()
    passes its text on as it is, with no type name put in front."""


def describe_error(e: Exception) -> str:
    """A failed tool call as short text: what went wrong, not a traceback. Shared by the agent
    loop and the MCP server, so the model reads the same words from both."""
    if isinstance(e, ToolFailed):
        text = str(e)
    elif isinstance(e, ValidationError):  # bad arguments from the model: say which field and why
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors(include_url=False)
        )
        text = f"invalid arguments: {problems}"
    elif isinstance(e, subprocess.CalledProcessError):  # str(e) would quote the whole ffmpeg command
        text = f"{Path(e.cmd[0]).name} failed with exit code {e.returncode}"
    else:
        text = f"{type(e).__name__}: {e}"
    notes = getattr(e, "__notes__", [])  # ffmpeg.run() attaches ffmpeg's own error message here
    if notes:
        text += " | " + " ".join(notes)[-MAX_ERROR_CHARS:]
    return text[:MAX_ERROR_CHARS]
