"""Shortsmith as an MCP server: the Phase 3 tools, served to any app that speaks MCP.

    host (Claude Desktop, our own agent) --JSON-RPC over stdio--> this process --> agents/tools.py

The host starts this file as a child process and talks to it with JSON-RPC messages, one
per line, on its stdin and stdout. It asks what the server offers (tools/list,
resources/list, prompts/list) and then uses it (tools/call, resources/read, prompts/get).
No code here knows which host is on the other end: that is the point of a standard.

A server offers three kinds of things, told apart by who decides to use them:
- tools: the MODEL decides to call them (generate_image, ...): actions, with effects.
- resources: the APP or the user decides to read them (a run's script.json): read-only context.
- prompts: the USER picks them (a "make a short video" template), like a slash command.

The host normally starts the server itself (the Claude Desktop config is in the Phase 4
build notes). By hand, from the repo root:
    uv run --env-file .env python -m shortsmith.mcp_server
stdout is the JSON-RPC channel: nothing else may be written there, so logs go to stderr.
"""

import argparse
import inspect
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from pydantic import BaseModel, Field, ValidationError

from shortsmith import llm
from shortsmith.agents import tools
from shortsmith.media import images
from shortsmith.models import HOOK_MAX_WORDS, NARRATION_WORDS, SCENES_MAX, SCENES_MIN, SCENES_PLANNED, Script

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = Path("runs")  # relative, like everywhere else: main() moves into --root first
RUN_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"  # a folder name, never a path: no "..", no slashes

log = logging.getLogger("shortsmith.mcp")

# Demo switch (--fail-first), as in Phase 3. A server process lives across many calls, so
# "already failed once" is remembered here, not in the per-call ToolContext.
FAIL_FIRST: set[str] = set()
FAILED: set[str] = set()

mcp = MCPServer(
    name="shortsmith",
    instructions=(
        "Makes ~30-second vertical short videos (YouTube Shorts, Reels, TikTok). "
        "Flow: (optional research: search_trends for what is getting views now, retrieve_hooks for "
        "proven hook patterns and brand rules) -> create_run with a script you write (or write_script) -> "
        "generate_image and text_to_speech for every scene -> (optional and PAID with real media: animate_image for "
        "ONE hero scene, only after the user agrees to the cost) -> render_video, which saves final.mp4 in the run folder."
    ),
)

RunId = Annotated[str, Field(
    pattern=RUN_ID_PATTERN, description="The run's id, as returned by create_run or write_script.",
)]


def run_dir(run_id: str) -> Path:
    """The folder of an existing run. The id is checked here too: resources get it from a URI."""
    if not re.fullmatch(RUN_ID_PATTERN, run_id) or not (RUNS / run_id / "script.json").exists():
        raise ValueError(f"no run {run_id!r}: start one with create_run or write_script")
    return RUNS / run_id


def new_run_dir() -> Path:
    """A fresh runs/<timestamp> folder; two runs started in the same second get -2, -3, ..."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for n in range(1, 100):
        folder = RUNS / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            folder.mkdir(parents=True)
            return folder
        except FileExistsError:
            continue
    raise RuntimeError("could not make a new run folder")


# --- the registry's tools, published again (media tools with a run_id) ---------------------


def registry_tool(t: tools.Tool) -> Callable[..., BaseModel]:
    """The MCP version of a registry tool: the same function and arguments, plus run_id for a
    media or clip tool. (Research tools, Phase 5, work on no run and are published as they are.)

    In Phase 3 the program bound one run to every call (ToolContext). A server serves many
    conversations, so the run comes with each call, and the context is loaded from disk each
    time: nothing is lost when the host restarts the server, and two hosts can each run their
    own server process on the same runs/ folder.

    MCPServer builds a tool's input schema from its function's signature, so this function
    gets one made from the registry's args model: one source for both front-ends.
    """
    needs_run = t.group != "research"  # media and clip tools work inside a run

    def call(**arguments: Any) -> BaseModel:
        try:
            ctx = None
            if needs_run:
                ctx = tools.ToolContext.for_run(run_dir(arguments.pop("run_id")), fail_first=FAIL_FIRST)
                ctx.failed = FAILED  # shared on purpose: "fails once" means once per server process
            return tools.run_tool(ctx, t.name, json.dumps(arguments))
        except Exception as e:  # a ToolError's text reaches the model; any other exception's text stays on the server
            raise ToolError(tools.describe_error(e)) from e

    params = [inspect.Parameter("run_id", inspect.Parameter.KEYWORD_ONLY, annotation=RunId)] if needs_run else []
    for name, field in t.args.model_fields.items():
        default = inspect.Parameter.empty if field.is_required() else field.default
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY,
                                        annotation=Annotated[field.annotation, field], default=default))
    returns = inspect.signature(t.fn).return_annotation  # ImageResult, ...: becomes the tool's outputSchema
    call.__signature__ = inspect.Signature(params, return_annotation=returns)  # type: ignore[attr-defined]
    call.__name__ = t.name
    return call


for _tool in tools.TOOLS.values():
    mcp.add_tool(registry_tool(_tool), name=_tool.name, description=_tool.description)


# --- starting a run: what a chat app needs before the media tools ----------------------------

_TARGETS = ", ".join(f"{low}-{high} in {llm.LANGUAGES[code].split(' (')[0]}" for code, (low, high) in NARRATION_WORDS.items())

CREATE_RUN = f"""Start a new video from a finished script you wrote. Returns the run_id the other tools need.

The script is checked before anything is made: {SCENES_MIN}-{SCENES_MAX} scenes ({SCENES_PLANNED} is best) with \
idx 1, 2, 3...; hook at most 8 words (rejected over {HOOK_MAX_WORDS}), and scene 1's narration starts with it; \
narration is spoken words only, about 30 seconds in total: {_TARGETS} words for all scenes together; the last \
scene's narration ends with the cta; title, hook, narration and cta in the script's language; visual_prompt in \
English, one concrete vertical 9:16 image, ending with the same style words in every scene; motion is zoom_in, \
zoom_out, pan_left or pan_right; duration_s is your estimate of each scene's speech. A rejected script comes \
back with the reason: fix that and call again.

Next: generate_image and text_to_speech for every scene (they don't depend on each other), then render_video."""


class RunStarted(BaseModel):
    run_id: str
    folder: str  # absolute, so the user can find final.mp4
    title: str
    language: str
    scenes: int


class ScriptWritten(RunStarted):
    script: Script  # the host shows it to the user; the narration is what will be spoken


def start_run(script: Script) -> RunStarted:
    folder = new_run_dir()
    (folder / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    log.info("new run %s: %r (%s, %d scenes)", folder.name, script.title, script.language, len(script.scenes))
    return RunStarted(run_id=folder.name, folder=str(folder.resolve()), title=script.title,
                      language=script.language, scenes=len(script.scenes))


@mcp.tool(description=CREATE_RUN)
def create_run(script: Script) -> RunStarted:
    # MCPServer has already validated `script` against the Script model (word counts and all)
    # before this runs: an invalid script comes back to the model as an error result.
    return start_run(script)


@mcp.tool()
def write_script(topic: str, language: Literal["en", "uz", "ru"] = "en") -> ScriptWritten:
    """Let Shortsmith's own writer model write the script for a topic, and start a run with it.
    Use this instead of create_run when you don't want to write the script yourself. Takes 10-60 s.
    Returns the run_id and the script. Next: generate_image and text_to_speech for every scene, then render_video."""
    try:
        script, _ = llm.write_script(topic, language)  # one LLM call, recorded in the ledger as "writer"
    except ValidationError as e:  # the writer model broke a rule; the call is still paid and in the ledger
        raise ToolError(f"the writer's script broke the rules, try again: {tools.describe_error(e)}") from e
    return ScriptWritten(**start_run(script).model_dump(), script=script)


# --- resources: read-only context the app (not the model) chooses to read ---------------------


@mcp.resource("shortsmith://runs", mime_type="application/json",
              description="Every run, newest first: id, title, language, scenes and whether final.mp4 exists.")
def runs_list() -> str:
    found = []
    for script_file in sorted(RUNS.glob("*/script.json"), reverse=True):
        try:
            script = Script.load(script_file)
        except (ValidationError, ValueError):
            continue  # an early experiment's script: not a run these tools can work on
        found.append({"run_id": script_file.parent.name, "title": script.title, "language": script.language,
                      "scenes": len(script.scenes), "final": (script_file.parent / "final.mp4").exists()})
    return json.dumps(found, ensure_ascii=False, indent=2)


@mcp.resource("shortsmith://runs/{run_id}/script", mime_type="application/json",
              description="A run's script.json: title, hook, scenes (narration, visual prompt, motion), cta, language.")
def run_script(run_id: str) -> str:
    try:
        return (run_dir(run_id) / "script.json").read_text(encoding="utf-8")
    except ValueError as e:
        raise ResourceNotFoundError(str(e)) from e


@mcp.resource("shortsmith://runs/{run_id}/status", mime_type="application/json",
              description="What a run already has: scenes with an image, scenes with a voiceover, and whether final.mp4 exists.")
def run_status(run_id: str) -> str:
    try:
        return tools.run_status(tools.ToolContext.for_run(run_dir(run_id))).model_dump_json()
    except ValueError as e:
        raise ResourceNotFoundError(str(e)) from e


# --- prompts: templates the user picks in the host -------------------------------------------


@mcp.prompt(title="Make a short video")
def make_short(topic: str, language: Literal["en", "uz", "ru"] = "en") -> str:
    """Make a ~30-second vertical short video about a topic: script, images, voice, final.mp4."""
    return f"""Make a ~30-second vertical short video about: {topic}
Language: {llm.LANGUAGES[language]} (language code '{language}').

0. Research first: search_trends (2-5 keywords) shows what is getting views on YouTube now, and
   retrieve_hooks gives hook patterns that fit the topic. Use one pattern for the hook.
1. Write the script yourself and start the run with create_run. Write it as this brief says:
---
{llm.WRITER_SYSTEM_PROMPT}
---
   If create_run rejects the script, fix what the error says and call it again.
   (If you'd rather not write it, call write_script and Shortsmith's own writer will.)
2. Call generate_image and text_to_speech for every scene. They don't depend on each other.
3. Call render_video. Then tell me the title, the hook, and the full path of final.mp4."""


# --- running it -------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortsmith's MCP server over stdio. An MCP host starts it.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="the folder that holds runs/ (default: the repo)")
    parser.add_argument("--fail-first", action="append", choices=sorted(tools.names("media")), default=[],
                        help="demo: this tool's first call fails on purpose (repeatable)")
    args = parser.parse_args()

    # A host starts the server from its own folder (Claude Desktop: its install folder), and
    # runs/, the ledger and ffmpeg's inputs are relative paths. So move to the repo first.
    os.chdir(args.root)
    FAIL_FIRST.update(args.fail_first)
    # One plain line per message on stderr (the host saves it to a log file). MCPServer set up
    # a rich console handler, which wraps every line into a narrow column when it isn't a terminal.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # While serving, the SDK points fd 1 at stderr, so a stray print() can't reach the channel.
    # But Python keeps print() output in a memory buffer when stdout is a pipe, and whatever
    # is still there at exit is written after the SDK has put the channel back (Phase 4 build
    # notes, E2). Line buffering sends every printed line to fd 1 at once: to stderr.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    log.info("shortsmith MCP server | root %s | images: %s | LLM provider: %s (key %s) | ffmpeg: %s%s",
             Path.cwd(), "placeholders (MOCK_MEDIA)" if images.mock_media() else "FLUX, PAID",
             os.environ.get("LLM_PROVIDER", "openai"), "set" if os.environ.get(llm.provider().key_env) else "missing",
             shutil.which("ffmpeg") or "NOT FOUND on PATH: render_video will fail",
             f" | first call fails on purpose: {', '.join(args.fail_first)}" if args.fail_first else "")
    mcp.run("stdio")  # blocks; returns when the host closes our stdin


if __name__ == "__main__":
    main()
