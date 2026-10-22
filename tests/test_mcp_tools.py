"""agents/mcp_tools.py: the bridge that lets the Phase 3 loop use the MCP server's tools.

The server runs in-process (the SDK's Client connects to it directly), the LLM is the
scripted fake from test_loop, voice and ffmpeg are faked: no subprocess, no network.
"""

import json
import os
import sys

import mcp_types
import pytest

from shortsmith import mcp_server
from shortsmith.agents import loop, mcp_tools, tools
from shortsmith.agents.loop import run_loop
from test_loop import ScriptedLLM, answer, image
from test_mcp_server import a_run, en_script, fake_media


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOCK_MEDIA", "1")
    monkeypatch.setattr(mcp_server, "FAILED", set())
    with mcp_tools.McpToolbox.connect(mcp_server.mcp) as connected:
        yield connected


# --- schemas -------------------------------------------------------------------------------


def test_the_model_sees_the_phase_3_schemas_with_run_id_hidden(box):
    over_mcp = box.schemas(tools.TOOLS, hidden=["run_id"])
    direct = tools.schemas()
    for mine, theirs in zip(over_mcp, direct, strict=True):
        theirs["function"]["parameters"].pop("additionalProperties")  # MCPServer doesn't forbid extra arguments
        assert mine == theirs


def test_a_tool_without_model_arguments_has_an_empty_object_schema(box):
    [render] = box.schemas(["render_video"], hidden=["run_id"])
    assert render["function"]["parameters"] == {"type": "object", "properties": {}}


def test_a_tool_the_server_lacks_is_a_key_error(box):
    with pytest.raises(KeyError):
        box.schemas(["make_coffee"])


# --- results -------------------------------------------------------------------------------


def test_a_structured_result_becomes_the_same_json_a_direct_call_returns():
    result = mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text="{...}")],
                                      structured_content={"scene": 1, "path": "scenes/scene_01.png", "reused": False})
    assert mcp_tools.tool_message(result) == '{"ok": true, "scene": 1, "path": "scenes/scene_01.png", "reused": false}'


def test_an_error_result_is_raised_and_the_loop_reports_its_text_unchanged():
    result = mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text="Error executing tool x: boom")],
                                      is_error=True)
    with pytest.raises(tools.ToolFailed) as failed:
        mcp_tools.tool_message(result)
    assert json.loads(loop.error_result(failed.value)) == {"ok": False, "error": "Error executing tool x: boom"}


def test_a_plain_text_result_is_passed_on():
    result = mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text="done")])
    assert mcp_tools.tool_message(result) == "done"


# --- calls ---------------------------------------------------------------------------------


def test_the_program_binds_the_run_and_the_model_cannot_change_it(box, tmp_path):
    a_run(tmp_path / "runs", "mine")
    a_run(tmp_path / "runs", "other")
    call_tool = box.tool_caller(run_id="mine")

    call_tool("generate_image", json.dumps({"scene": 1, "run_id": "other"}))
    assert (tmp_path / "runs" / "mine" / "scenes" / "scene_01.png").exists()
    assert not (tmp_path / "runs" / "other" / "scenes").exists()


def test_arguments_that_are_not_a_json_object_raise_like_a_direct_call(box):
    call_tool = box.tool_caller(run_id="r1")
    with pytest.raises(json.JSONDecodeError):
        call_tool("generate_image", "{scene: 1")
    with pytest.raises(ValueError, match="JSON object"):
        call_tool("generate_image", "[1]")


def test_the_phase_3_loop_reaches_final_mp4_through_mcp_unchanged(box, tmp_path, monkeypatch):
    """The same scripted model as test_loop's producer test: renders too early, reads the error,
    makes everything, renders, finishes. Only the tools moved behind MCP."""
    fake_media(monkeypatch)
    run_id = a_run(tmp_path / "runs")
    everything = [image(i) for i in range(1, 5)] + [("text_to_speech", {"scene": i}) for i in range(1, 5)]
    model = ScriptedLLM(answer(calls=[("render_video", {})]), answer(calls=everything),
                        answer(calls=[("render_video", {})]), answer("final.mp4 is ready."))

    result = run_loop([{"role": "user", "content": "Make the video."}], box.schemas(tools.TOOLS, hidden=["run_id"]),
                      model, box.tool_caller(run_id=run_id))

    too_early = json.loads(model.requests[1][-1]["content"])
    assert too_early == {"ok": False, "error": "Error executing tool render_video: FileNotFoundError: not ready to "
                                               "render: no image for scenes [1, 2, 3, 4]; no voiceover for scenes [1, 2, 3, 4]"}
    voice = json.loads(model.requests[2][-1]["content"])
    assert voice == {"ok": True, "scene": 4, "path": "audio/scene_04.mp3", "duration_s": 4.9, "words": 15}
    assert json.loads(result.messages[-2]["content"])["path"] == "final.mp4"
    assert (result.stop_reason, result.turns) == ("done", 4)
    assert (tmp_path / "runs" / run_id / "final.mp4").exists()


def test_the_cli_runs_the_producer_over_mcp(tmp_path, monkeypatch, capsys):
    """`loop --from <run> --mcp`, with the server in-process and the model scripted: the program
    starts the run with create_run and reads the script and status as resources."""
    fake_media(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOCK_MEDIA", "1")
    monkeypatch.setattr(mcp_server, "FAILED", set())
    source = tmp_path / "source"
    source.mkdir()
    (source / "script.json").write_text(json.dumps(en_script()), encoding="utf-8")
    everything = [image(i) for i in range(1, 7)] + [("text_to_speech", {"scene": i}) for i in range(1, 7)]
    model = ScriptedLLM(answer(calls=everything), answer(calls=[("render_video", {})]), answer("Done."))
    monkeypatch.setattr(mcp_tools, "server_params", lambda fail_first=(): mcp_server.mcp)
    monkeypatch.setattr(loop, "llm_caller", lambda model_name=None: model)
    monkeypatch.setattr("sys.argv", ["loop", "--from", str(source), "--mcp"])

    loop.main()

    [run_dir] = (tmp_path / "runs").iterdir()
    assert (run_dir / "final.mp4").exists() and (run_dir / "agent_log.jsonl").exists()
    task = model.requests[0][1]["content"]
    assert task.startswith("Script: 'A title' (en, 6 scenes)") and "images for scenes none" in task
    assert [s["function"]["name"] for s in model.schemas[0]] == ["generate_image", "text_to_speech", "render_video"]
    printed = capsys.readouterr().out
    assert "over MCP: server 'shortsmith', protocol 2025-11-25" in printed and "stopped: done after 3 turns" in printed


def test_the_server_gets_only_the_environment_it_needs(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MEDIA", "1")
    monkeypatch.setenv("SOMETHING_ELSE", "x")
    params = mcp_tools.server_params(root=tmp_path, fail_first=["render_video"])

    assert params.command == sys.executable
    assert params.args == ["-m", "shortsmith.mcp_server", "--root", str(tmp_path), "--fail-first", "render_video"]
    assert params.env["MOCK_MEDIA"] == "1" and "SOMETHING_ELSE" not in params.env
    assert set(params.env) <= set(mcp_tools.PASS_ENV) and os.environ.get("SOMETHING_ELSE") == "x"
