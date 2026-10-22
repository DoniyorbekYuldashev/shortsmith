"""Checks for the Step 3.2 exercise: the same loop tests as tests/test_loop.py.

A scripted fake LLM plays back prepared answers in order and keeps a copy of every request,
so each test can check exactly what the model would have been sent on each turn. Tools are
faked too: nothing here needs an API key, the internet, or ffmpeg.
Run from the repo root:  uv run pytest exercises/phase-3
"""

import copy
import itertools
import json

import pytest

from agent_loop import run_loop  # pytest puts this folder on sys.path
from shortsmith.llm import ChatResult, ToolCall

SCHEMAS = [{"type": "function", "function": {"name": "generate_image", "parameters": {"type": "object"}}}]
SIGNATURE = {"google": {"thought_signature": "EmAKXgFpFH0T...opaque"}}  # what Gemini 3 attaches
_ids = itertools.count(1)


def answer(text: str = "", calls: list[tuple[str, dict]] | tuple = (), prompt_tokens: int = 100,
           completion_tokens: int = 20) -> ChatResult:
    """One prepared answer, shaped like Gemini's: the first tool call carries a thought signature."""
    tool_calls = [ToolCall(id=f"call_{next(_ids)}", name=name, arguments=json.dumps(args)) for name, args in calls]
    message = {"role": "assistant"} | ({"content": text} if text else {})
    if tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            | ({"extra_content": SIGNATURE} if i == 0 else {})
            for i, c in enumerate(tool_calls)
        ]
    return ChatResult(text=text, tool_calls=tool_calls, message=message, model="fake", prompt_tokens=prompt_tokens,
                      completion_tokens=completion_tokens, cached_tokens=0, cost_usd=0.001, latency_s=0.1)


class ScriptedLLM:
    def __init__(self, *answers: ChatResult):
        self.answers = list(answers)
        self.requests: list[list[dict]] = []  # a deep copy of `messages` at each call
        self.schemas: list = []

    def __call__(self, messages, tool_schemas):
        self.requests.append(copy.deepcopy(messages))
        self.schemas.append(tool_schemas)
        if not self.answers:
            raise AssertionError("the loop asked for more turns than were scripted")
        return self.answers.pop(0)


class FakeTools:
    """call_tool stand-in: records every call; each name in `fail` raises once, like a flaky service."""

    def __init__(self, fail: set[str] = frozenset()):
        self.calls: list[tuple[str, dict]] = []
        self.fail = set(fail)

    def __call__(self, name: str, arguments: str) -> str:
        self.calls.append((name, json.loads(arguments)))
        if name in self.fail:
            self.fail.discard(name)
            raise ConnectionError("the service closed the connection")
        if name not in {"generate_image", "text_to_speech", "render_video"}:
            raise ValueError(f"unknown tool {name!r}")
        return json.dumps({"ok": True, "tool": name, **json.loads(arguments)})


def start() -> list[dict]:
    return [{"role": "system", "content": "You are a producer."}, {"role": "user", "content": "Make the video."}]


def image(scene: int) -> tuple[str, dict]:
    return ("generate_image", {"scene": scene})


# --- the loop ------------------------------------------------------------------------------


def test_a_plain_answer_ends_the_loop_at_once():
    model = ScriptedLLM(answer("Nothing to do."))
    result = run_loop(start(), SCHEMAS, model, FakeTools())

    assert (result.stop_reason, result.turns, result.text) == ("done", 1, "Nothing to do.")
    assert result.messages[-1] == {"role": "assistant", "content": "Nothing to do."}


def test_the_model_sees_the_tool_schemas_on_every_turn():
    model = ScriptedLLM(answer(calls=[image(1)]), answer("Done."))
    run_loop(start(), SCHEMAS, model, FakeTools())
    assert model.schemas == [SCHEMAS, SCHEMAS]  # the API is stateless: tools are sent again each time


def test_a_tool_result_goes_back_with_the_id_of_its_call():
    model = ScriptedLLM(answer(calls=[image(2)]), answer("Done."))
    fake = FakeTools()
    run_loop(start(), SCHEMAS, model, fake)

    assert fake.calls == [("generate_image", {"scene": 2})]  # OUR code ran it, with the model's arguments
    *_, asked, result = model.requests[1]
    assert result == {
        "role": "tool",
        "tool_call_id": asked["tool_calls"][0]["id"],
        "content": '{"ok": true, "tool": "generate_image", "scene": 2}',
    }


def test_the_assistant_message_goes_back_unchanged_thought_signature_included():
    first = answer(calls=[image(1), image(2)])
    model = ScriptedLLM(first, answer("Done."))
    run_loop(start(), SCHEMAS, model, FakeTools())

    sent_back = model.requests[1][2]
    assert sent_back == first.message  # rebuilt from ToolCall objects, it would lose extra_content
    assert sent_back["tool_calls"][0]["extra_content"] == SIGNATURE


def test_parallel_calls_are_all_run_and_each_gets_its_own_answer_in_order():
    calls = [image(1), image(2), ("text_to_speech", {"scene": 1})]
    model = ScriptedLLM(answer(calls=calls), answer("Done."))
    fake = FakeTools()
    run_loop(start(), SCHEMAS, model, fake)

    assert fake.calls == [(name, args) for name, args in calls]
    history = model.requests[1]
    asked_ids = [c["id"] for c in history[2]["tool_calls"]]
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool", "tool", "tool"]
    assert [m["tool_call_id"] for m in history[3:]] == asked_ids


def test_a_failing_tool_is_reported_to_the_model_which_can_retry():
    retry = ("text_to_speech", {"scene": 2})
    model = ScriptedLLM(answer(calls=[retry]), answer(calls=[retry]), answer("Done."))
    result = run_loop(start(), SCHEMAS, model, FakeTools(fail={"text_to_speech"}))

    failed = json.loads(model.requests[1][-1]["content"])
    assert failed == {"ok": False, "error": "ConnectionError: the service closed the connection"}
    assert json.loads(model.requests[2][-1]["content"])["ok"] is True
    assert result.stop_reason == "done"


def test_an_unknown_tool_is_reported_to_the_model_too():
    model = ScriptedLLM(answer(calls=[("make_coffee", {"sugar": 2})]), answer("Sorry."))
    run_loop(start(), SCHEMAS, model, FakeTools())

    error = json.loads(model.requests[1][-1]["content"])["error"]
    assert error == "ValueError: unknown tool 'make_coffee'"


def test_max_steps_stops_a_model_that_never_finishes():
    model = ScriptedLLM(*[answer(calls=[image(1)]) for _ in range(10)])
    result = run_loop(start(), SCHEMAS, model, FakeTools(), max_steps=3)

    assert (result.stop_reason, result.turns, len(model.requests)) == ("max_steps", 3, 3)
    assert result.messages[-1]["role"] == "tool"  # the last call was still answered: the history stays valid


def test_the_token_budget_stops_the_loop():
    model = ScriptedLLM(*[answer(calls=[image(1)], prompt_tokens=300, completion_tokens=100) for _ in range(10)])
    result = run_loop(start(), SCHEMAS, model, FakeTools(), max_tokens=1000)

    assert (result.stop_reason, result.turns) == ("token_budget", 3)  # 400, 800, then 1200 >= 1000
    assert result.prompt_tokens + result.completion_tokens == 1200


def test_a_final_answer_is_done_even_when_it_crosses_the_budget():
    model = ScriptedLLM(answer("Done.", prompt_tokens=5000))
    assert run_loop(start(), SCHEMAS, model, FakeTools(), max_tokens=1000).stop_reason == "done"


def test_every_turn_is_logged_and_the_totals_add_up():
    model = ScriptedLLM(answer(calls=[image(1)], prompt_tokens=100), answer(calls=[image(2)], prompt_tokens=150),
                        answer("Done.", prompt_tokens=200))
    logged = []
    result = run_loop(start(), SCHEMAS, model, FakeTools(), log=logged.append)

    assert [t.turn for t in logged] == [1, 2, 3]
    assert [t.stop for t in logged] == [None, None, "done"]
    assert [t.tokens_so_far for t in logged] == [120, 290, 510]
    assert logged[0].tool_calls[0].name == "generate_image" and logged[0].tool_calls[0].ok
    assert (result.prompt_tokens, result.completion_tokens) == (450, 60)
    assert result.cost_usd == pytest.approx(0.003)
