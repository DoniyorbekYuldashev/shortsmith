"""🧠 Exercise, Step 4.4 (optional): the bridge between MCP and the Chat Completions API.

Our agent loop speaks Chat Completions (tool schemas in the "function" format, tool results
as plain strings); the MCP server speaks MCP (inputSchema, CallToolResult with isError).
These two small functions translate between them. The finished versions are chat_schema()
and tool_message() in src/shortsmith/agents/mcp_tools.py: try these first, then compare.
Check them from the repo root:  uv run pytest exercises/phase-4
No server, key or internet needed: the tests build the MCP objects by hand.
"""

from collections.abc import Iterable

import mcp_types

from shortsmith.agents import tools  # given: tools._without_titles(), tools.ok_json(), tools.ToolFailed
from shortsmith.llm import Message


def chat_schema(tool: mcp_types.Tool, hidden: Iterable[str] = ()) -> Message:
    """An MCP tool (from tools/list) in the Chat Completions "function" format.

    tool.name, tool.description (may be None) and tool.input_schema (a dict: a JSON schema with
    "properties", "required", ...) are what you need. Look at a real one in the build notes
    (the tools/list answer) or in the tests.

    Hints:
    1. Work on a copy of tool.input_schema (copy.deepcopy): the tool object must not change.
    2. For each name in `hidden` (e.g. "run_id": the program fills it in, the model mustn't
       see it): remove it from "properties" and from "required" if it is there.
    3. If "required" is now an empty list, remove the key (the API is happy without it).
    4. Remove Pydantic's "title" keys with tools._without_titles(...).
    5. Return {"type": "function", "function": {"name": ..., "description": <"" if None>,
       "parameters": <your cleaned schema>}}.
    """
    raise NotImplementedError("your turn: follow the hints above")


def tool_message(result: mcp_types.CallToolResult) -> str:
    """An MCP tool result as the content string of a Chat Completions "tool" message.

    result.content is a list of content blocks; the text ones are mcp_types.TextContent with a
    .text. result.is_error says whether the tool failed. result.structured_content is a dict
    (or None) when the tool declared an output schema.

    Hints:
    1. Join the .text of every TextContent block in result.content with "\\n".
    2. If result.is_error: raise tools.ToolFailed(<that text>). (The loop catches it and sends
       {"ok": false, "error": <text>} to the model, exactly like a failing direct call.)
    3. Else, if result.structured_content is not None: return tools.ok_json(result.structured_content)
       (that adds "ok": true, the same JSON the direct call_tool returns).
    4. Else return the joined text.
    """
    raise NotImplementedError("your turn: follow the hints above")
