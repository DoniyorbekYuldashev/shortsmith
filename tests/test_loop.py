"""The agent loop, driven by a scripted fake LLM: no API, no key, no real media.

ScriptedLLM plays back prepared answers in order and keeps a copy of every request, so a
test can check exactly what the model would have been sent on each turn. The same tests
(minus the last section) check the exercise in exercises/phase-3.
"""

import copy
import itertools
import json
import subprocess
from pathlib import Path

import httpx
import openai
import pytest
from pydantic import ValidationError

from shortsmith import llm
from shortsmith.agents import loop, tools
from shortsmith.agents.loop import run_loop
from shortsmith.llm import ChatResult, ToolCall
from shortsmith.media import ffmpeg

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


# --- the real code only: error text, the log file, the paced LLM, the whole producer ----------


def test_bad_arguments_become_a_short_error_that_names_the_field():
    with pytest.raises(ValidationError) as error:
        tools.SceneArgs.model_validate_json('{"scene": "two"}')
    text = json.loads(loop.error_result(error.value))["error"]
    assert text == "invalid arguments: scene: Input should be a valid integer, unable to parse string as an integer"


def test_an_ffmpeg_failure_shows_ffmpegs_message_not_the_whole_command():
    error = subprocess.CalledProcessError(1, ["ffmpeg"] + ["-i", "x" * 50] * 40)
    error.add_note("ffmpeg said: subtitles.srt: No such file or directory")
    result = loop.error_result(error)

    assert json.loads(result) == {
        "ok": False, "error": "ffmpeg failed with exit code 1 | ffmpeg said: subtitles.srt: No such file or directory"
    }


def test_a_long_error_is_cut_short():
    result = json.loads(loop.error_result(RuntimeError("x" * 5000)))
    assert len(result["error"]) == loop.MAX_ERROR_CHARS


def test_the_journal_writes_one_json_line_per_turn(tmp_path, capsys):
    path = tmp_path / "agent_log.jsonl"
    model = ScriptedLLM(answer(calls=[image(1), image(2)]), answer("Done."))
    run_loop(start(), SCHEMAS, model, FakeTools(fail={"generate_image"}), log=loop.journal(path))

    first, last = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert first["turn"] == 1 and first["stop"] is None
    assert [(c["name"], c["arguments"], c["ok"]) for c in first["tool_calls"]] == [
        ("generate_image", '{"scene": 1}', False), ("generate_image", '{"scene": 2}', True)
    ]
    assert {"prompt_tokens", "completion_tokens", "tokens_so_far", "latency_s", "cost_usd", "ts"} <= set(first)
    assert last["stop"] == "done" and last["text"] == "Done."
    printed = capsys.readouterr().out
    assert "generate_image(1) FAILED, generate_image(2)" in printed and "STOP: done" in printed


def _rate_limited() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test/chat/completions"))
    return openai.RateLimitError("RESOURCE_EXHAUSTED", response=response, body=None)


def test_the_llm_caller_waits_out_a_rate_limit(monkeypatch):
    outcomes = [_rate_limited(), answer("ok")]

    def fake_chat_tools(messages, tool_schemas, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm, "chat_tools", fake_chat_tools)
    waits = []
    call = loop.llm_caller(min_interval_s=4.5, sleep=waits.append)  # fake sleep: the clock doesn't move

    assert call(start(), SCHEMAS).text == "ok"
    assert waits[0] == loop.RATE_LIMIT_WAIT_S  # backed off after the 429...
    assert 4.0 < waits[1] <= 4.5  # ...then still kept the pace before trying again


def test_the_llm_caller_gives_up_after_its_retries(monkeypatch):
    attempts = []

    def always_limited(*args, **kwargs):
        attempts.append(1)
        raise _rate_limited()

    monkeypatch.setattr(llm, "chat_tools", always_limited)
    with pytest.raises(openai.RateLimitError):
        loop.llm_caller(sleep=lambda s: None)(start(), SCHEMAS)
    assert len(attempts) == loop.RATE_LIMIT_RETRIES


def test_the_producer_reaches_final_mp4_with_the_real_tools(tmp_path, monkeypatch):
    """Real registry and placeholder images; voice and ffmpeg faked; the model is scripted to
    render too early first, so it gets the 'not ready' error and has to fix that."""
    from test_tools import _script, fake_synthesize

    def fake_assemble(run_dir: Path, script, voiceovers) -> Path:
        (run_dir / "final.mp4").write_bytes(b"mp4")
        return run_dir / "final.mp4"

    monkeypatch.setenv("MOCK_MEDIA", "1")
    monkeypatch.setattr(tools.tts, "synthesize", fake_synthesize)
    monkeypatch.setattr(tools.ffmpeg, "assemble", fake_assemble)
    monkeypatch.setattr(tools.ffmpeg, "probe", lambda path: ffmpeg.VideoInfo(
        width=1080, height=1920, fps=30, frames=900, duration_s=30.0, size_bytes=2_500_000,
        video_codec="h264", pix_fmt="yuv420p", audio_codec="aac"))
    (tmp_path / "script.json").write_text(_script().model_dump_json(), encoding="utf-8")
    ctx = tools.ToolContext.for_run(tmp_path)

    everything = [image(i) for i in range(1, 5)] + [("text_to_speech", {"scene": i}) for i in range(1, 5)]
    model = ScriptedLLM(answer(calls=[("render_video", {})]), answer(calls=everything),
                        answer(calls=[("render_video", {})]), answer("final.mp4 is ready."))
    result = run_loop(
        [{"role": "user", "content": loop.task_message(ctx.script, tools.run_status(ctx))}], tools.schemas(), model,
        lambda name, arguments: tools.call_tool(ctx, name, arguments),
    )

    too_early = json.loads(model.requests[1][-1]["content"])
    assert too_early == {"ok": False, "error": "FileNotFoundError: not ready to render: no image for scenes "
                                               "[1, 2, 3, 4]; no voiceover for scenes [1, 2, 3, 4]"}
    assert json.loads(result.messages[-2]["content"])["path"] == "final.mp4"
    assert (result.stop_reason, result.turns) == ("done", 4)
    assert (tmp_path / "final.mp4").exists() and (tmp_path / "voiceovers.json").exists()


def test_the_task_message_lists_the_scenes_and_what_already_exists(tmp_path):
    from test_tools import _script

    (tmp_path / "script.json").write_text(_script().model_dump_json(), encoding="utf-8")
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "scene_03.png").write_bytes(b"png")
    ctx = tools.ToolContext.for_run(tmp_path)
    text = loop.task_message(ctx.script, tools.run_status(ctx))

    assert text.startswith("Script: 't' (en, 4 scenes)\nScene 1: word word")
    assert "images for scenes [3]; voiceovers for scenes none; final.mp4 not yet" in text
    assert "prompt 1" not in text  # visual prompts stay in the run folder: the tools read them


def test_paid_media_needs_an_explicit_flag(monkeypatch):
    monkeypatch.setenv("MOCK_MEDIA", "0")
    monkeypatch.setattr("sys.argv", ["loop", "--run", "runs/nowhere"])
    with pytest.raises(SystemExit):
        loop.main()
