"""The agent's tools through MCP: the same loop, with the tools now in another process.

    run_loop --call_tool()--> McpToolbox --JSON-RPC over stdio--> mcp_server.py --> agents/tools.py

run_loop() never knew where its tools live. It gets the schemas the model sees and a
call_tool(name, arguments) function. Here both come from an MCP server: the schemas from
tools/list, and every call becomes a tools/call request. The loop doesn't change at all.

The MCP client is async (it has to read the server's stdout while it waits), the loop is
not. An anyio "blocking portal" runs the client's event loop in a background thread; each
call from the loop is handed to that thread and waited for, like a normal function call.
"""

import copy
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mcp_types
from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import Client, StdioServerParameters

from shortsmith.agents import tools
from shortsmith.llm import Message

REPO_ROOT = Path(__file__).resolve().parents[3]

# stdio_client starts the server with only a few "safe" variables of ours (PATH, APPDATA, ...),
# not the whole environment. These are the ones the server needs, passed on by name.
PASS_ENV = ("MOCK_MEDIA", "LLM_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY", "FAL_KEY",
            "VECTOR_STORE", "DATABASE_URL")  # the RAG store, for retrieve_hooks


def server_params(root: Path = REPO_ROOT, fail_first: Iterable[str] = ()) -> StdioServerParameters:
    """How our agent starts the server: the module Claude Desktop starts, as our own child process."""
    flags = [part for name in fail_first for part in ("--fail-first", name)]
    return StdioServerParameters(
        command=sys.executable,  # this venv's Python: uv already set it up for us
        args=["-m", "shortsmith.mcp_server", "--root", str(root), *flags],
        env={key: os.environ[key] for key in PASS_ENV if key in os.environ},
        cwd=root,
    )


def chat_schema(tool: mcp_types.Tool, hidden: Iterable[str] = ()) -> Message:
    """An MCP tool in the Chat Completions "function" format the LLM API takes.

    Arguments in `hidden` are left out because the program fills them in (the run_id, as
    Phase 3's ToolContext did), so the model can't point a call at another run. Pydantic's
    "title" keys go too, as in Phase 3: they tell the model nothing and cost tokens every turn.
    """
    params = copy.deepcopy(tool.input_schema)
    for name in hidden:
        params.get("properties", {}).pop(name, None)
        if name in params.get("required", []):
            params["required"].remove(name)
    if params.get("required") == []:
        del params["required"]  # Phase 3's render_video schema had none either
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": tools._without_titles(params)},
    }


def tool_message(result: mcp_types.CallToolResult) -> str:
    """An MCP tool result as the text of a Chat Completions tool message: the same JSON a
    direct call returns.

    MCP reports failure with a flag (isError); a tool message is only a string. So a failure is
    raised as ToolFailed, and the loop turns it into {"ok": false, "error": ...} like any other.
    """
    text = "\n".join(block.text for block in result.content if isinstance(block, mcp_types.TextContent))
    if result.is_error:
        raise tools.ToolFailed(text)
    if result.structured_content is not None:
        return tools.ok_json(result.structured_content)
    return text


class McpToolbox:
    """A connected MCP client with a blocking (non-async) face, for run_loop and main()."""

    def __init__(self, portal: BlockingPortal, client: Client):
        self.portal = portal
        self.client = client

    @classmethod
    @contextmanager
    def connect(cls, server: Any, mode: str = "legacy") -> Iterator["McpToolbox"]:
        """Start the server (or attach to an in-process one in tests), shake hands, close on exit.

        mode="legacy": the initialize handshake of protocol 2025-11-25, the version hosts have
        spoken so far. The SDK's default ("auto") negotiates the newer 2026-07-28 protocol when
        the server has it (ours does), which has no handshake; the build notes show both.
        """
        with start_blocking_portal() as portal, portal.wrap_async_context_manager(Client(server, mode=mode)) as client:
            yield cls(portal, client)

    def list_tools(self) -> list[mcp_types.Tool]:
        return self.portal.call(self.client.list_tools).tools

    def call(self, name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
        return self.portal.call(self.client.call_tool, name, arguments)

    def read_text(self, uri: str) -> str:
        """A text resource. The app decides to read it, not the model: no tool call involved."""
        contents = self.portal.call(self.client.read_resource, uri).contents
        return "".join(c.text for c in contents if isinstance(c, mcp_types.TextResourceContents))

    def schemas(self, names: Iterable[str], hidden: Iterable[str] = ()) -> list[Message]:
        """The named tools, as the model will see them. A name the server lacks raises KeyError."""
        listed = {t.name: t for t in self.list_tools()}
        return [chat_schema(listed[name], hidden) for name in names]

    def tool_caller(self, **bound: Any) -> Callable[[str, str], str]:
        """call_tool for run_loop: the model's JSON arguments plus the bound ones, sent over MCP.
        Raises (ToolFailed, bad JSON) like the direct call_tool, so the loop handles both alike."""

        def call_tool(name: str, arguments: str) -> str:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            return tool_message(self.call(name, args | bound))  # bound last: the model can't override them

        return call_tool
