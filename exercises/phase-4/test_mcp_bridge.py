"""Checks for the Step 4.4 exercise: MCP tool schemas and results, translated for the agent loop.

The MCP objects below are built by hand, with the exact schemas our server publishes
(copied from the real tools/list answer in the build notes). Run from the repo root:
    uv run pytest exercises/phase-4
"""

import copy
import json

import mcp_types
import pytest

from mcp_bridge import chat_schema, tool_message  # pytest puts this folder on sys.path
from shortsmith.agents import tools

GENERATE_IMAGE = mcp_types.Tool(
    name="generate_image",
    description="Make the vertical 9:16 image for one scene, from the script's visual prompt.",
    input_schema={
        "type": "object", "title": "generate_imageArguments", "required": ["run_id", "scene"],
        "properties": {
            "run_id": {"type": "string", "title": "Run Id", "pattern": "^[A-Za-z0-9_-]{1,64}$",
                       "description": "The run's id, as returned by create_run or write_script."},
            "scene": {"type": "integer", "title": "Scene", "minimum": 1,
                      "description": "Scene number (idx) from the script, starting at 1."},
        },
    },
)
RENDER_VIDEO = mcp_types.Tool(
    name="render_video", description=None,
    input_schema={"type": "object", "title": "render_videoArguments", "required": ["run_id"],
                  "properties": {"run_id": {"type": "string", "title": "Run Id"}}},
)


def text_result(text: str, **fields) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=text)], **fields)


# --- chat_schema ----------------------------------------------------------------------------


def test_the_hidden_argument_is_gone_and_the_rest_is_the_phase_3_schema():
    assert chat_schema(GENERATE_IMAGE, hidden=["run_id"]) == {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Make the vertical 9:16 image for one scene, from the script's visual prompt.",
            "parameters": {"type": "object", "required": ["scene"], "properties": {
                "scene": {"type": "integer", "minimum": 1, "description": "Scene number (idx) from the script, starting at 1."}
            }},
        },
    }


def test_nothing_is_hidden_unless_asked():
    params = chat_schema(GENERATE_IMAGE)["function"]["parameters"]
    assert params["required"] == ["run_id", "scene"]
    assert set(params["properties"]) == {"run_id", "scene"}
    assert "title" not in json.dumps(params)


def test_an_empty_required_list_is_dropped_and_a_missing_description_is_empty():
    function = chat_schema(RENDER_VIDEO, hidden=["run_id"])["function"]
    assert function["parameters"] == {"type": "object", "properties": {}}
    assert function["description"] == ""


def test_the_mcp_tool_is_not_changed():
    before = copy.deepcopy(GENERATE_IMAGE.input_schema)
    chat_schema(GENERATE_IMAGE, hidden=["run_id"])
    assert GENERATE_IMAGE.input_schema == before


def test_hiding_an_argument_the_tool_does_not_have_is_fine():
    assert chat_schema(GENERATE_IMAGE, hidden=["nope"])["function"]["parameters"]["required"] == ["run_id", "scene"]


# --- tool_message ---------------------------------------------------------------------------


def test_a_structured_result_becomes_the_direct_calls_json():
    result = text_result("{...}", structured_content={"scene": 1, "path": "scenes/scene_01.png", "reused": False})
    assert tool_message(result) == '{"ok": true, "scene": 1, "path": "scenes/scene_01.png", "reused": false}'


def test_an_error_result_is_raised_as_tool_failed_with_its_text():
    with pytest.raises(tools.ToolFailed) as failed:
        tool_message(text_result("Error executing tool render_video: not ready", is_error=True))
    assert str(failed.value) == "Error executing tool render_video: not ready"


def test_an_error_wins_even_if_there_is_structured_content():
    with pytest.raises(tools.ToolFailed):
        tool_message(text_result("boom", is_error=True, structured_content={"scene": 1}))


def test_plain_text_blocks_are_joined():
    result = mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text="line 1"),
                                               mcp_types.TextContent(type="text", text="line 2")])
    assert tool_message(result) == "line 1\nline 2"
