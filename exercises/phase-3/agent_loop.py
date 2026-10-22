"""🧠 Exercise, Step 3.2: the agent loop.

The finished version is run_loop() in src/shortsmith/agents/loop.py: try this one first,
then compare. Check it from the repo root:  uv run pytest exercises/phase-3
The tests drive your loop with a scripted fake LLM, so no API key or internet is needed.
"""

from collections.abc import Callable

from shortsmith.agents.loop import (  # the data types and the error helper are given
    MAX_STEPS,  # 12
    MAX_TOKENS,  # 60_000
    AgentResult,
    CallLLM,
    CallLog,
    CallTool,
    StopReason,
    Turn,
    error_result,
)
from shortsmith.llm import Message


def run_loop(
    messages: list[Message],
    tool_schemas: list[Message],
    call_llm: CallLLM,
    call_tool: CallTool,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
    log: Callable[[Turn], None] | None = None,
) -> AgentResult:
    """Run the agent until it answers without a tool call, or a budget stops it.

    - call_llm(messages, tool_schemas) -> ChatResult: one LLM call. Useful fields: .text,
      .tool_calls (a list of ToolCall: .id, .name, .arguments as a JSON string), .message (the
      assistant message as a dict), .prompt_tokens, .completion_tokens, .total_tokens,
      .cost_usd, .latency_s.
    - call_tool(name, arguments) -> str: runs one tool and returns short JSON. It MAY RAISE.

    Hints, in order:
    1. Keep a running token count (`used`) and a list of the Turns so far.
    2. Loop: result = call_llm(messages, tool_schemas); add result.total_tokens to `used`.
    3. Append result.message to `messages` exactly as it is. Don't rebuild it from
       result.tool_calls: Gemini puts a "thought signature" in it and rejects the next request
       without it (the build notes show the real 400 error).
    4. For EVERY call in result.tool_calls, in order (several in one answer = parallel calls):
       - content = call_tool(call.name, call.arguments)
       - if that raises (any Exception), content = error_result(e) instead: the error goes to
         the model as the result, and the loop keeps going
       - append {"role": "tool", "tool_call_id": call.id, "content": content}
       - keep a CallLog(id=..., name=..., arguments=..., result=content, ok=..., seconds=...)
    5. Decide whether to stop, checking in this order:
       - no tool calls at all             -> "done" (the model answered in words)
       - used >= max_tokens               -> "token_budget"
       - this was turn number max_steps   -> "max_steps"
       Notice: the tool calls of the last turn were still run and answered, so the history
       stays valid (every tool call has its tool message).
    6. Build Turn(turn=<1, 2, ...>, text=..., tool_calls=<your CallLogs>, prompt_tokens=...,
       completion_tokens=..., tokens_so_far=used, cost_usd=..., latency_s=..., stop=<reason or None>)
       and call log(turn) if a log was given.
    7. When stopping, return AgentResult(stop_reason=..., text=result.text, turns=<how many>,
       prompt_tokens=<sum over turns>, completion_tokens=<sum>, cost_usd=<sum>, messages=messages).
    """
    raise NotImplementedError("your turn: follow the hints above")
