"""The agent loop, by hand: what "an agent" is once the frameworks are taken away.

    messages + tool schemas -> model -> tool calls? -> OUR code runs them -> results appended -> again

The model never executes anything. It answers with the names and JSON arguments of the
calls it wants; this code validates and runs them, and sends each result back as a "tool"
message. The loop ends when the model answers without a tool call, or when a budget
(LLM turns, tokens) runs out.

Demo, from the repo root (copies a run's script.json into a new run folder first):
    uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855
The same, with the tools behind the MCP server (Phase 4) instead of imported:
    uv run --env-file .env python -m shortsmith.agents.loop --from runs/2026-10-22_112855 --mcp
"""

import argparse
import json
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import openai
from pydantic import BaseModel, Field

from shortsmith import ledger, llm
from shortsmith.agents import mcp_tools, tools
from shortsmith.llm import ChatResult, Message
from shortsmith.media import images
from shortsmith.models import Script

StopReason = Literal["done", "max_steps", "token_budget"]
CallLLM = Callable[[list[Message], list[Message]], ChatResult]  # (messages, tool schemas) -> one answer
CallTool = Callable[[str, str], str]  # (name, JSON arguments) -> short JSON result; may raise

MAX_STEPS = 12  # LLM turns. The happy path needs 3-4; a model stuck retrying must not run forever
MAX_TOKENS = 60_000  # prompt + completion over the whole run. The happy path used 3,440 (build notes, demo 1):
# room for many retries, but a stuck model can't run up a bill (each turn re-sends the whole history)
MAX_ERROR_CHARS = tools.MAX_ERROR_CHARS  # an error result is read by the model and paid for on every later turn


class CallLog(BaseModel):
    id: str
    name: str
    arguments: str  # JSON text exactly as the model wrote it
    result: str  # exactly what went back to the model
    ok: bool
    seconds: float


class Turn(BaseModel):
    """One LLM call plus the tool calls it asked for: one line of agent_log.jsonl."""

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn: int
    text: str  # what the model said besides its tool calls (often nothing)
    tool_calls: list[CallLog]
    prompt_tokens: int  # grows every turn: the whole history is sent again
    completion_tokens: int
    tokens_so_far: int  # what the token budget is checked against
    cost_usd: float
    latency_s: float  # the LLM call only; each tool's time is in its CallLog
    stop: StopReason | None = None  # set on the last turn only


class AgentResult(BaseModel):
    stop_reason: StopReason
    text: str  # the model's last words
    turns: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    messages: list[Message]  # the whole conversation, ready to inspect or continue


def error_result(e: Exception) -> str:
    """A failed tool call as the short JSON the model reads: what went wrong, not a traceback.
    (The text itself comes from tools.describe_error, which the MCP server uses too.)"""
    return json.dumps({"ok": False, "error": tools.describe_error(e)}, ensure_ascii=False)


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

    `messages` (system + user to begin with) is extended in place: afterwards it holds the
    whole conversation. A tool that raises does not end the loop: its error goes back to
    the model as the result, so the model can retry or change course. Every tool call in a
    turn is answered before the loop stops, so the history is always valid to continue.
    """
    turns: list[Turn] = []
    used = 0
    while True:
        result = call_llm(messages, tool_schemas)
        used += result.total_tokens
        messages.append(result.message)  # unchanged: Gemini checks its thought signatures on the next turn

        calls = []
        for call in result.tool_calls:  # several calls in one answer = parallel calls: run all, answer each
            start = time.perf_counter()
            try:
                content, ok = call_tool(call.name, call.arguments), True
            except Exception as e:  # any tool failure is feedback for the model, not a crash
                content, ok = error_result(e), False
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            calls.append(CallLog(id=call.id, name=call.name, arguments=call.arguments, result=content, ok=ok,
                                 seconds=round(time.perf_counter() - start, 2)))

        stop: StopReason | None = None
        if not result.tool_calls:
            stop = "done"  # the model answered in words: it considers the job finished
        elif used >= max_tokens:
            stop = "token_budget"  # checked before paying for another call; it can overshoot by one turn
        elif len(turns) + 1 >= max_steps:
            stop = "max_steps"
        turn = Turn(turn=len(turns) + 1, text=result.text, tool_calls=calls, prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens, tokens_so_far=used, cost_usd=result.cost_usd,
                    latency_s=result.latency_s, stop=stop)
        turns.append(turn)
        if log:
            log(turn)
        if stop:
            return AgentResult(
                stop_reason=stop,
                text=result.text,
                turns=len(turns),
                prompt_tokens=sum(t.prompt_tokens for t in turns),
                completion_tokens=sum(t.completion_tokens for t in turns),
                cost_usd=sum(t.cost_usd for t in turns),
                messages=messages,
            )


# --- the real LLM, paced for the free tier --------------------------------------------------

MIN_INTERVAL_S = 4.5  # Gemini's free tier allows ~15 requests a minute: one every 4 s, plus a margin
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_WAIT_S = 20  # x attempt: 20, 40, 60 s. A per-minute limit clears within a minute


def llm_caller(
    model: str | None = None,
    ledger_path: Path = ledger.DEFAULT_PATH,
    min_interval_s: float = MIN_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> CallLLM:
    """chat_tools() for the loop: at most one request every `min_interval_s`, and a 429 (rate
    limit, RESOURCE_EXHAUSTED on Gemini) waits and tries again instead of ending the run.
    A refused request isn't billed, so only the call that succeeds lands in the ledger."""
    last = float("-inf")

    def call(messages: list[Message], tool_schemas: list[Message]) -> ChatResult:
        nonlocal last
        for attempt in range(1, RATE_LIMIT_RETRIES + 1):
            wait = last + min_interval_s - time.monotonic()
            if wait > 0:
                sleep(wait)
            last = time.monotonic()
            try:
                return llm.chat_tools(messages, tool_schemas, model=model, step="agent", ledger_path=ledger_path)
            except openai.RateLimitError as e:
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                print(f"  429 rate limit ({str(e)[:120]}...): waiting {RATE_LIMIT_WAIT_S * attempt} s")
                sleep(RATE_LIMIT_WAIT_S * attempt)
        raise AssertionError("unreachable")

    return call


# --- the demo: a Producer agent that turns an existing script into final.mp4 --------------

PRODUCER_SYSTEM_PROMPT = """You are the producer of a short vertical video. The script is already written; \
your job is to turn it into final.mp4 using your tools.

- Every scene needs an image and a voiceover before the video can be rendered.
- You may ask for several tool calls at once when they don't depend on each other.
- Tools answer with JSON. If one returns "ok": false, read the error and decide: retry, fix the cause, or stop.
- When final.mp4 is rendered (or it clearly can't be), reply with one short sentence and call no tool."""


def task_message(script: Script, status: tools.RunStatus) -> str:
    """The user message: the script in compact form (no visual prompts: the tools read those
    from the run folder) and what already exists, so a half-finished run isn't redone."""
    lines = [f"Script: {script.title!r} ({script.language}, {len(script.scenes)} scenes)"]
    lines += [f"Scene {scene.idx}: {scene.narration}" for scene in script.scenes]
    lines.append(
        f"\nAlready made: images for scenes {status.images or 'none'}; voiceovers for scenes "
        f"{status.voiceovers or 'none'}; final.mp4 {'exists' if status.final else 'not yet'}."
    )
    return "\n".join(lines)


def _short_call(call: CallLog) -> str:
    """generate_image(2) for the console; the model's arguments may not even be valid JSON."""
    try:
        args = json.loads(call.arguments or "{}")
        shown = ",".join(str(v) for v in args.values()) if isinstance(args, dict) else call.arguments
    except json.JSONDecodeError:
        shown = call.arguments
    return f"{call.name}({shown}){'' if call.ok else ' FAILED'}"


def journal(log_path: Path) -> Callable[[Turn], None]:
    """Append each turn to agent_log.jsonl as it happens (a crash keeps the turns so far) and print it."""

    def write(turn: Turn) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(turn.model_dump_json() + "\n")
        calls = ", ".join(_short_call(c) for c in turn.tool_calls) or f"answer: {turn.text.strip()!r}"
        print(f"turn {turn.turn:>2} | {turn.prompt_tokens:>5} in {turn.completion_tokens:>4} out "
              f"| {turn.latency_s:>5.1f} s | {calls}" + (f" | STOP: {turn.stop}" if turn.stop else ""))

    return write


def produce(run_dir: Path, script: Script, status: tools.RunStatus, tool_schemas: list[Message],
            call_tool: CallTool, args: argparse.Namespace, via: str) -> None:
    """What both ways of reaching the tools share: the prompt, the loop, the log files, the summary."""
    messages: list[Message] = [
        {"role": "system", "content": PRODUCER_SYSTEM_PROMPT},
        {"role": "user", "content": task_message(script, status)},
    ]
    print(f"{run_dir} | {script.title!r} | tools: {', '.join(s['function']['name'] for s in tool_schemas)} | {via}"
          + (f" | first call fails on purpose: {', '.join(args.fail_first)}" if args.fail_first else ""))

    start, cost_before = time.perf_counter(), ledger.total_cost(ledger.DEFAULT_PATH)
    result = run_loop(
        messages,
        tool_schemas,
        call_llm=llm_caller(args.model),
        call_tool=call_tool,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        log=journal(run_dir / "agent_log.jsonl"),
    )
    (run_dir / "agent_messages.json").write_text(json.dumps(result.messages, indent=2, ensure_ascii=False), encoding="utf-8")

    final = run_dir / "final.mp4"
    print(
        f"\nstopped: {result.stop_reason} after {result.turns} turn{'s' * (result.turns != 1)} | {result.prompt_tokens} in + "
        f"{result.completion_tokens} out tokens | ledger ${ledger.total_cost(ledger.DEFAULT_PATH) - cost_before:.6f} "
        f"list price | {time.perf_counter() - start:.0f} s\n"
        f"{final if final.exists() else 'no final.mp4'} | log: {run_dir / 'agent_log.jsonl'}"
    )


def produce_over_mcp(args: argparse.Namespace) -> None:
    """The same producer, but every tool call goes to the MCP server (the one Claude Desktop uses).

    The program, not the model, starts the run (create_run) and reads the script and the
    run's status (resources); the model only sees the three media tools, with run_id filled in.
    """
    start = time.perf_counter()
    with mcp_tools.McpToolbox.connect(mcp_tools.server_params(fail_first=args.fail_first)) as box:
        ready_s = time.perf_counter() - start
        if args.source:
            script = Script.load(args.source / "script.json")
            try:
                created = mcp_tools.tool_message(box.call("create_run", {"script": script.model_dump(mode="json")}))
            except tools.ToolFailed as e:
                raise SystemExit(f"create_run rejected {args.source / 'script.json'}: {e}")
            run_id = json.loads(created)["run_id"]
        else:
            run_id = args.run.name  # a folder in the server's runs/
        script = Script.model_validate_json(box.read_text(f"shortsmith://runs/{run_id}/script"), context={"saved": True})
        status = tools.RunStatus.model_validate_json(box.read_text(f"shortsmith://runs/{run_id}/status"))
        via = (f"over MCP: server '{box.client.server_info.name}', protocol {box.client.protocol_version}, "
               f"started in {ready_s:.1f} s")
        produce(Path("runs") / run_id, script, status, box.schemas(tools.TOOLS, hidden=["run_id"]),
                box.tool_caller(run_id=run_id), args, via)


def main() -> None:
    parser = argparse.ArgumentParser(description="An agent turns a run's script.json into final.mp4 (Phase 3).")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--run", type=Path, help="work in this run folder (it must have script.json)")
    where.add_argument("--from", dest="source", type=Path, help="copy this run's script.json into a new run folder")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="LLM turns allowed")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="prompt + completion tokens allowed")
    parser.add_argument("--fail-first", action="append", choices=sorted(tools.TOOLS), default=[],
                        help="demo: this tool's first call fails on purpose (repeatable)")
    parser.add_argument("--model", help="default: the provider's default model")
    parser.add_argument("--allow-paid-media", action="store_true", help="needed when MOCK_MEDIA=0 (FLUX costs money)")
    parser.add_argument("--mcp", action="store_true", help="use the tools through the MCP server (Phase 4)")
    args = parser.parse_args()
    if not images.mock_media() and not args.allow_paid_media:
        parser.error("MOCK_MEDIA=0 means paid images: add --allow-paid-media if you really want that")
    if args.mcp:
        produce_over_mcp(args)
        return

    if args.source:
        run_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir.mkdir(parents=True)
        shutil.copy(args.source / "script.json", run_dir / "script.json")
    else:
        run_dir = args.run
    ctx = tools.ToolContext.for_run(run_dir, fail_first=set(args.fail_first))
    produce(run_dir, ctx.script, tools.run_status(ctx), tools.schemas(),
            lambda name, arguments: tools.call_tool(ctx, name, arguments), args, via="direct calls")


if __name__ == "__main__":
    main()
