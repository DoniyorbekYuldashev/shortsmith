"""mcp_server.py through a real MCP client: the official SDK's Client, connected to the server
in-process (the same requests and results as over stdio, without a subprocess), plus two
tests that start the server as a real subprocess and talk to it over stdin/stdout.

Placeholder images are real; voice and ffmpeg are faked, so nothing here needs the network.
"""

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from mcp import MCPError

from shortsmith import llm, mcp_server
from shortsmith.agents import mcp_tools, tools
from shortsmith.media import ffmpeg
from shortsmith.models import Script
from test_tools import _script, fake_synthesize

MEDIA_TOOLS = {"generate_image", "text_to_speech", "render_video"}


def en_script(**changes) -> dict:
    """A script that passes every rule for fresh scripts (6 scenes, ~80 English words)."""
    scenes = [{"idx": i, "narration": " ".join(["word"] * 13), "visual_prompt": f"prompt {i}, warm film look",
               "duration_s": 5, "motion": "zoom_in"} for i in range(1, 7)]
    return {"title": "A title", "hook": "Stop scrolling now", "scenes": scenes, "cta": "Follow", "language": "en"} | changes


@pytest.fixture
def runs(tmp_path, monkeypatch) -> Path:
    """The server works in runs/ under the current folder: a temporary one here."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOCK_MEDIA", "1")
    monkeypatch.setattr(mcp_server, "FAIL_FIRST", set())
    monkeypatch.setattr(mcp_server, "FAILED", set())
    return tmp_path / "runs"


@pytest.fixture
def box(runs):
    with mcp_tools.McpToolbox.connect(mcp_server.mcp) as connected:
        yield connected


def a_run(runs: Path, name: str = "r1") -> str:
    (runs / name).mkdir(parents=True)
    (runs / name / "script.json").write_text(_script().model_dump_json(), encoding="utf-8")
    return name


def text(result) -> str:
    return result.content[0].text


def fake_media(monkeypatch) -> None:
    def fake_assemble(run_dir: Path, script, voiceovers) -> Path:
        (run_dir / "final.mp4").write_bytes(b"mp4")
        return run_dir / "final.mp4"

    monkeypatch.setattr(tools.tts, "synthesize", fake_synthesize)
    monkeypatch.setattr(tools.ffmpeg, "assemble", fake_assemble)
    monkeypatch.setattr(tools.ffmpeg, "probe", lambda path: ffmpeg.VideoInfo(
        width=1080, height=1920, fps=30, frames=900, duration_s=30.0, size_bytes=2_500_000,
        video_codec="h264", pix_fmt="yuv420p", audio_codec="aac"))


# --- what the server offers --------------------------------------------------------------


def test_the_server_offers_the_registry_tools_plus_two_that_start_a_run(box):
    listed = {t.name: t for t in box.list_tools()}
    assert set(listed) == MEDIA_TOOLS | {"create_run", "write_script"}
    for name in MEDIA_TOOLS:  # descriptions come from the Phase 3 docstrings, not a copy
        assert listed[name].description == tools.TOOLS[name].description


def test_a_media_tool_takes_the_registry_arguments_plus_a_run_id(box):
    schema = {t.name: t for t in box.list_tools()}["generate_image"].input_schema
    assert schema["required"] == ["run_id", "scene"]
    assert schema["properties"]["run_id"]["pattern"] == mcp_server.RUN_ID_PATTERN  # a folder name, never a path
    registry = tools.schema(tools.TOOLS["generate_image"])["function"]["parameters"]
    assert tools._without_titles(schema)["properties"]["scene"] == registry["properties"]["scene"]
    assert {t.name: t for t in box.list_tools()}["render_video"].input_schema["required"] == ["run_id"]


def test_a_media_tool_declares_its_result_model_as_output_schema(box):
    output = {t.name: t for t in box.list_tools()}["text_to_speech"].output_schema
    assert set(output["properties"]) == set(tools.VoiceResult.model_fields)


def test_create_run_takes_the_script_model_as_its_input_schema(box):
    create = {t.name: t for t in box.list_tools()}["create_run"]
    assert create.input_schema["required"] == ["script"]
    assert set(create.input_schema["$defs"]) == {"Script", "Scene"}
    assert "78-84 in English" in create.description  # the rules the validator checks are in the description


# --- starting a run ----------------------------------------------------------------------


def test_create_run_saves_a_valid_script_and_returns_its_run_id(box, runs):
    result = box.call("create_run", {"script": en_script()})

    assert not result.is_error
    started = result.structured_content
    assert (started["title"], started["language"], started["scenes"]) == ("A title", "en", 6)
    saved = Script.load(runs / started["run_id"] / "script.json")
    assert saved.hook == "Stop scrolling now"
    assert Path(started["folder"]).is_absolute()  # the user has to find final.mp4


def test_two_runs_started_in_the_same_second_get_different_folders(box):
    first = box.call("create_run", {"script": en_script()}).structured_content["run_id"]
    second = box.call("create_run", {"script": en_script()}).structured_content["run_id"]
    assert first != second


@pytest.mark.parametrize("changes, reason", [
    ({"hook": "one two three four five six seven eight nine ten eleven"}, "hook has 11 words, max 10"),
    ({"scenes": en_script()["scenes"][:3]}, "3 scenes, expected 4-8"),
    ({"language": "uz"}, "narration has 78 words in total, expected about 55-60"),
])
def test_create_run_rejects_a_script_that_breaks_the_rules_with_the_reason(box, runs, changes, reason):
    result = box.call("create_run", {"script": en_script(**changes)})

    assert result.is_error
    assert reason in text(result)
    assert not runs.exists() or not any(runs.iterdir())  # nothing was started


def test_write_script_uses_shortsmiths_writer_and_starts_a_run(box, runs, monkeypatch):
    asked = {}

    def fake_writer(topic, language):
        asked["args"] = (topic, language)
        return Script.model_validate(en_script()), None

    monkeypatch.setattr(llm, "write_script", fake_writer)
    result = box.call("write_script", {"topic": "Registan at night", "language": "en"})

    assert asked["args"] == ("Registan at night", "en")
    assert result.structured_content["script"]["hook"] == "Stop scrolling now"
    assert (runs / result.structured_content["run_id"] / "script.json").exists()


def test_a_writer_script_that_breaks_the_rules_is_an_error_result(box, monkeypatch):
    def broken_writer(topic, language):
        Script.model_validate(en_script(scenes=[]))  # raises like chat_json does on a bad answer

    monkeypatch.setattr(llm, "write_script", broken_writer)
    result = box.call("write_script", {"topic": "x"})
    assert result.is_error and "the writer's script broke the rules" in text(result)


# --- the media tools over MCP --------------------------------------------------------------


def test_a_media_tool_answers_with_the_same_json_as_a_direct_call(box, runs):
    run_id = a_run(runs)
    result = box.call("generate_image", {"run_id": run_id, "scene": 2})

    assert result.structured_content == {"scene": 2, "path": "scenes/scene_02.png", "reused": False}
    image = runs / run_id / "scenes" / "scene_02.png"
    assert image.exists()
    image.unlink()  # so the direct call makes it again instead of reusing it
    direct = tools.call_tool(tools.ToolContext.for_run(runs / run_id), "generate_image", '{"scene": 2}')
    assert mcp_tools.tool_message(result) == direct  # what the model reads is byte for byte the same


def test_voice_survives_a_server_restart_and_the_video_renders(runs, monkeypatch):
    fake_media(monkeypatch)
    run_id = a_run(runs)
    with mcp_tools.McpToolbox.connect(mcp_server.mcp) as first:
        for idx in (1, 2):
            assert not first.call("text_to_speech", {"run_id": run_id, "scene": idx}).is_error
    assert not (runs / run_id / "voiceovers.json").exists()  # only half voiced: nothing in memory to lose

    with mcp_tools.McpToolbox.connect(mcp_server.mcp) as second:  # "restarted": a fresh client and per-call contexts
        assert json.loads(second.read_text(f"shortsmith://runs/{run_id}/status"))["voiceovers"] == [1, 2]
        for idx in (3, 4):
            second.call("text_to_speech", {"run_id": run_id, "scene": idx})
        for idx in range(1, 5):
            second.call("generate_image", {"run_id": run_id, "scene": idx})
        result = second.call("render_video", {"run_id": run_id})

    assert result.structured_content == {"path": "final.mp4", "duration_s": 30.0, "size_mb": 2.5, "scenes": 4}
    assert (runs / run_id / "voiceovers.json").exists()  # complete now: `make --run` can re-edit it


def test_a_failing_tool_is_an_error_result_with_the_same_words_as_phase_3(box, runs):
    run_id = a_run(runs)
    result = box.call("render_video", {"run_id": run_id})

    assert result.is_error
    assert text(result) == ("Error executing tool render_video: FileNotFoundError: not ready to render: "
                            "no image for scenes [1, 2, 3, 4]; no voiceover for scenes [1, 2, 3, 4]")


@pytest.mark.parametrize("arguments, words", [
    ({"run_id": "nope", "scene": 1}, "no run 'nope': start one with create_run or write_script"),
    ({"run_id": "../../etc", "scene": 1}, "String should match pattern"),  # no path tricks
    ({"run_id": "r1", "scene": 0}, "Input should be greater than or equal to 1"),
    ({"run_id": "r1", "scene": 9}, "no scene 9: this script has scenes 1-4"),
])
def test_bad_calls_are_error_results_the_model_can_read(box, runs, arguments, words):
    a_run(runs)
    result = box.call("generate_image", arguments)
    assert result.is_error and words in text(result)


def test_fail_first_fails_once_per_server_process(box, runs, monkeypatch):
    monkeypatch.setattr(mcp_server, "FAIL_FIRST", {"generate_image"})
    run_id = a_run(runs)

    first = box.call("generate_image", {"run_id": run_id, "scene": 1})
    assert first.is_error and "ConnectionError" in text(first)
    assert not box.call("generate_image", {"run_id": run_id, "scene": 1}).is_error
    assert not box.call("generate_image", {"run_id": run_id, "scene": 2}).is_error


# --- resources and prompts -----------------------------------------------------------------


def test_the_runs_resource_lists_every_run(box, runs):
    a_run(runs, "2026-01-01_000000")
    a_run(runs, "2026-01-02_000000")
    (runs / "2026-01-02_000000" / "final.mp4").write_bytes(b"mp4")

    assert [r.uri for r in box.portal.call(box.client.list_resources).resources] == ["shortsmith://runs"]
    listed = json.loads(box.read_text("shortsmith://runs"))
    assert [(r["run_id"], r["final"]) for r in listed] == [("2026-01-02_000000", True), ("2026-01-01_000000", False)]


def test_a_runs_script_and_status_are_resource_templates(box, runs):
    run_id = a_run(runs)
    box.call("generate_image", {"run_id": run_id, "scene": 3})

    templates = box.portal.call(box.client.list_resource_templates).resource_templates
    assert {t.uri_template for t in templates} == {"shortsmith://runs/{run_id}/script", "shortsmith://runs/{run_id}/status"}
    assert box.read_text(f"shortsmith://runs/{run_id}/script") == (runs / run_id / "script.json").read_text(encoding="utf-8")
    assert json.loads(box.read_text(f"shortsmith://runs/{run_id}/status")) == {"images": [3], "voiceovers": [], "final": False}


def test_a_missing_run_is_a_protocol_error_for_resources(box, runs):
    with pytest.raises(MCPError, match="no run 'nope'"):  # not an isError result: an app, not a model, asked
        box.read_text("shortsmith://runs/nope/script")


def test_the_prompt_is_a_template_with_the_writers_brief(box):
    prompts = box.portal.call(box.client.list_prompts).prompts
    assert [(p.name, [(a.name, a.required) for a in p.arguments]) for p in prompts] == [
        ("make_short", [("topic", True), ("language", False)])
    ]
    got = box.portal.call(box.client.get_prompt, "make_short", {"topic": "Registan at night", "language": "uz"})
    message = got.messages[0]
    assert message.role == "user"
    assert "about: Registan at night" in message.content.text and "Uzbek (Latin script)" in message.content.text
    assert llm.WRITER_SYSTEM_PROMPT in message.content.text and "create_run" in message.content.text


# --- the real thing: a subprocess on stdin/stdout -------------------------------------------


def test_the_server_works_over_stdio_as_a_subprocess(tmp_path):
    a_run(tmp_path / "runs")
    with mcp_tools.McpToolbox.connect(mcp_tools.server_params(root=tmp_path)) as box:
        assert {t.name for t in box.list_tools()} == MEDIA_TOOLS | {"create_run", "write_script"}
        result = box.call("generate_image", {"run_id": "r1", "scene": 1})
    assert result.structured_content["path"] == "scenes/scene_01.png"
    assert (tmp_path / "runs" / "r1" / "scenes" / "scene_01.png").exists()


NOISY_SERVER = """
import sys
from shortsmith import mcp_server
from shortsmith.agents import tools

real = tools.TOOLS["generate_image"].fn

def noisy(ctx, args):
    print("a stray print() inside a tool")
    return real(ctx, args)

tools.TOOLS["generate_image"] = tools.TOOLS["generate_image"].model_copy(update={"fn": noisy})
sys.argv = ["mcp_server", "--root", sys.argv[1]]
mcp_server.main()
"""


def test_stdout_carries_only_json_rpc_even_when_a_tool_prints(tmp_path):
    """Raw pipes, no SDK client: every line the server writes to stdout must be a JSON-RPC
    message, from the first request to after shutdown. The tool here prints on purpose."""
    a_run(tmp_path / "runs")
    server = subprocess.Popen([sys.executable, "-c", NOISY_SERVER, str(tmp_path)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=lambda: [lines.put(line) for line in server.stdout], daemon=True)
    reader.start()

    def send(message: dict) -> dict | None:
        server.stdin.write(json.dumps(message) + "\n")
        server.stdin.flush()
        return json.loads(lines.get(timeout=60)) if "id" in message else None  # fails on a line that isn't JSON

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        called = send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "generate_image", "arguments": {"run_id": "r1", "scene": 1}}})
    finally:
        server.stdin.close()  # closing stdin is how a stdio host shuts the server down
        stderr = server.stderr.read()
        server.wait(timeout=30)
        reader.join(timeout=10)  # every line the server ever wrote is in `lines` now

    assert listed["id"] == 2 and len(listed["result"]["tools"]) == 5
    assert called["id"] == 3 and called["result"]["structuredContent"]["path"] == "scenes/scene_01.png"
    leftover = []
    while not lines.empty():
        leftover.append(lines.get())
    assert leftover == []  # nothing more on stdout, not even at exit
    assert "a stray print() inside a tool" in stderr  # it went to the log, not into the channel
    assert "shortsmith MCP server | root" in stderr
