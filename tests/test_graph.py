"""agents/graph.py end to end, offline: a fake LLM (answers by response schema), fake voice and ffmpeg,
a real LangGraph with a real SQLite checkpointer in a temp folder."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionMessage
from PIL import Image

from shortsmith import ledger, llm, tracing
from shortsmith.agents import critic, graph, producer, writer
from shortsmith.agents.state import RunState, Usage, add_usage
from shortsmith.media import animate, ffmpeg, images, tts
from shortsmith.media.tts import Voiceover, Word
from shortsmith.models import CriticReport, ScriptChecks

WORDS13 = "one two three four five six seven eight nine ten eleven twelve thirteen"


def scene(i: int, words: str = WORDS13) -> dict:
    return {"idx": i, "narration": words if i > 1 else f"Stop scrolling now. {' '.join(words.split()[3:])}",
            "visual_prompt": f"picture {i}, warm light", "duration_s": 5, "motion": "zoom_in"}


def option(n: int, short: bool = False) -> dict:
    words = "too short" if short else WORDS13
    return {"title": f"Option {n}", "hook": "Stop scrolling now.", "cta": "thirteen",
            "language": "en", "scenes": [scene(i, words) for i in range(1, 7)]}


class FakeLLM:
    """Answers by the response schema's name; counts every request per schema."""

    def __init__(self, critic_scores: list[tuple[float, list[int]]]):
        self.critic_scores = list(critic_scores)
        self.calls: dict[str, int] = {}
        self.requests: list[dict] = []

    def answer(self, name: str, request: dict) -> dict:
        if name == "ScriptOptions":
            return {"options": [option(1), option(2), option(3, short=True)]}
        if name == "WriterPick":
            return {"scores": [{"option": 1, "hook": 6, "clarity": 6, "fit": 6},
                               {"option": 2, "hook": 8, "clarity": 7, "fit": 7}], "best": 2, "reason": "better hook"}
        if name == "DirectorPlan":
            return {"style": "cinematic, warm light", "hero_scene": 1, "hero_motion": "steam rises",
                    "hero_reason": "the hook", "shots": [{"idx": i, "visual_prompt": f"shot {i}", "motion": "pan_left"}
                                                         for i in range(1, 7)]}
        if name == "CriticVerdict":
            score, weak = self.critic_scores.pop(0) if len(self.critic_scores) > 1 else self.critic_scores[0]
            return {"score": score, "hook": "fine", "summary": "ok", "cannot_judge": "image quality (placeholders)",
                    "weak_scenes": [{"scene": w, "problem": "visual", "reason": f"scene {w} off topic",
                                     "fix": f"show the subject of scene {w}"} for w in weak]}
        if name == "SceneFixes":
            user = request["messages"][1]["content"]
            flagged = [int(part.split(",")[0]) for part in user.split("- scene ")[1:]]
            return {"hook": "Stop scrolling now.",
                    "scenes": [{"scene": w, "narration": "ignored", "visual_prompt": f"revised shot {w} ({self.calls[name]})"}
                               for w in flagged]}
        if name == "ResearchNotes":
            return {"trending": ["t"], "hooks": [{"pattern": "POV", "hook": "POV: you found it", "why": "w"}],
                    "angles": ["a"], "avoid": ["x"], "visual_style": ""}
        raise AssertionError(f"unexpected schema {name}")

    def create(self, **request):
        name = request["response_format"]["json_schema"]["name"]
        self.calls[name] = self.calls.get(name, 0) + 1
        self.requests.append(request)
        content = json.dumps(self.answer(name, request))
        return NS(model="gemini-3.5-flash-lite", usage=NS(prompt_tokens=100, completion_tokens=50, prompt_tokens_details=None),
                  choices=[NS(message=ChatCompletionMessage(role="assistant", content=content))])


def fake_synthesize(text: str, voice: str, out_path: Path) -> Voiceover:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"mp3")
    words = [Word(text=w, start_s=i * 0.3, end_s=i * 0.3 + 0.25) for i, w in enumerate(text.split())]
    return Voiceover(audio_path=out_path, duration_s=len(words) * 0.3 + 0.4, words=words)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A temp repo root (runs/, ledger, checkpoints) with every outside service faked."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOCK_MEDIA", "1")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv(graph.CRASH_ENV, raising=False)
    monkeypatch.setattr(llm, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(writer, "_library", {"library: Open loop (en)": "This cafe has no menu."})
    fake = FakeLLM([(5.0, [3]), (8.0, [])])
    monkeypatch.setattr(llm, "_client", lambda *_: NS(chat=NS(completions=NS(create=fake.create))))
    monkeypatch.setattr(tts, "synthesize", fake_synthesize)
    made = {"images": [], "clips": [], "edits": 0}

    def fake_image(prompt, path, label="", step="producer"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prompt.encode())
        made["images"].append(path.name)
        return path

    def fake_clip(image, motion, out, step="producer"):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"clip")
        made["clips"].append(out.name)
        return out

    def fake_assemble(run_dir, script, voiceovers, music=None):
        made["edits"] += 1
        (run_dir / "final.mp4").write_bytes(f"edit {made['edits']}: {[s.motion for s in script.scenes]}".encode())
        return run_dir / "final.mp4"

    def fake_frames(run_dir, script, voiceovers, focus, round_):
        folder = run_dir / "critic" / f"round_{round_}"
        folder.mkdir(parents=True, exist_ok=True)
        frames = []
        for idx in critic.pick_scenes([s.idx for s in script.scenes], focus):
            jpeg = folder / f"scene_{idx:02d}.jpg"
            Image.new("RGB", (8, 8), "red").save(jpeg)
            frames.append((idx, 1.0 * idx, jpeg))
        return frames

    monkeypatch.setattr(images, "generate_image", fake_image)
    monkeypatch.setattr(animate, "animate_image", fake_clip)
    monkeypatch.setattr(ffmpeg, "assemble", fake_assemble)
    monkeypatch.setattr(critic, "run_frames", fake_frames)
    return NS(llm=fake, made=made, db=tmp_path / "checkpoints" / "graph.sqlite", root=tmp_path)


def start(world, approve="yes", **kwargs) -> tuple[RunState, dict]:
    state = graph.new_state("coffee shop in Tashkent", **kwargs)
    with graph.open_graph(world.db) as app:
        values = graph.drive(app, state.model_dump(), state.run_id, approve=(lambda q: approve) if approve else None)
    return state, values


# --- the whole graph ---------------------------------------------------------------------------------


def test_topic_to_final_video_with_one_critic_loop(world):
    state, values = start(world)
    s = RunState.model_validate(values)

    assert Path(s.final_path).read_bytes().startswith(b"edit 2")  # edited, criticised, revised, edited again
    assert [r.score for r in s.critic_rounds] == [5.0, 8.0] and graph.stop_reason(s).startswith("passed")
    assert [(r.scene, r.changed) for r in s.revisions] == [(3, True)]
    assert s.plan.scenes[2].visual_prompt.startswith("revised shot 3")
    assert s.made == ["round 0: images 1, 2, 3, 4, 5, 6; voices 1, 2, 3, 4, 5, 6; clip 1",
                      "round 1: images 3"]  # only the weak scene's picture was made again; its voice didn't change
    assert (s.folder / "history" / "round_1" / "scenes" / "scene_03.png").exists()  # the old picture is kept
    assert len(s.script_options) == 2 and "option 3" in s.rejected_options[0] and s.chosen.title == "Option 2"
    assert world.llm.calls == {"ScriptOptions": 1, "WriterPick": 1, "DirectorPlan": 1, "CriticVerdict": 2, "SceneFixes": 1}
    assert {n: u.calls for n, u in s.usage.items()} == {
        "writer": 2, "director": 1, "approve_hero": 0, "producer": 0, "editor": 0, "critic": 2, "revise": 1}
    assert s.tokens_used == 6 * 150  # the summing reducer: 6 calls x (100 + 50)
    lines = ledger.entries(Path("runs/ledger.jsonl"), s.run_id)
    assert [e.step for e in lines] == ["writer", "writer-pick", "director", "critic", "revise", "critic"]


def test_the_critic_sees_the_regenerated_scene_in_the_next_round(world):
    _, values = start(world)
    s = RunState.model_validate(values)
    assert 3 in s.critic_rounds[1].frames
    vision = [r for r in world.llm.requests if r["response_format"]["json_schema"]["name"] == "CriticVerdict"][0]
    parts = vision["messages"][1]["content"]
    assert [p["type"] for p in parts].count("image_url") == 4
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_the_loop_stops_after_max_rounds(world):
    world.llm.critic_scores = [(4.0, [2, 3])]  # never satisfied
    _, values = start(world, max_rounds=2)
    s = RunState.model_validate(values)
    assert len(s.critic_rounds) == 3 and world.llm.calls["SceneFixes"] == 2
    assert graph.stop_reason(s).startswith("max rounds")
    assert world.made["edits"] == 3  # the editor re-renders after every round


def test_a_good_first_score_ends_the_run(world):
    world.llm.critic_scores = [(8.5, [])]
    _, values = start(world)
    s = RunState.model_validate(values)
    assert len(s.critic_rounds) == 1 and s.revisions == [] and "SceneFixes" not in world.llm.calls


def test_research_runs_first_when_asked(world, monkeypatch):
    from shortsmith.agents import researcher

    def fake_research(topic, language, brand):
        notes, result = llm.chat_json([{"role": "user", "content": topic}], researcher.ResearchNotes, step="researcher")
        return researcher.Research(topic=topic, language=language, brand=brand, trend_queries=[topic], trends=None,
                                   hooks=[], brand_rules=[], notes=notes, prompt_tokens=result.prompt_tokens,
                                   completion_tokens=result.completion_tokens, cost_usd=result.cost_usd, latency_s=0)

    monkeypatch.setattr(researcher, "research", fake_research)
    state, values = start(world, research=True)
    s = RunState.model_validate(values)
    assert s.research.hooks[0].pattern == "POV" and (s.folder / "research.json").exists()
    writer_request = [r for r in world.llm.requests if r["response_format"]["json_schema"]["name"] == "ScriptOptions"][0]
    assert "Hook patterns that fit" in writer_request["messages"][1]["content"]  # the notes reached the writer
    assert "POV: you found it" not in writer_request["messages"][1]["content"]  # v2: patterns without ready-made hooks


# --- routing -------------------------------------------------------------------------------------------


def report(score: float, weak: list[int], round_: int = 0) -> CriticReport:
    checks = ScriptChecks(words=[13] * 6, total_words=78, target=(78, 84), in_range=True, hook_words=3, hook_copied_from=None)
    return CriticReport(score=score, hook="h", summary="s", cannot_judge="c", round=round_, mode="video", frames=[1],
                        model="m", checks=checks, weak_scenes=[{"scene": w, "problem": "visual", "reason": "r", "fix": "f"}
                                                                for w in weak])


def routed(*reports: CriticReport, max_rounds: int = 2, pass_score: float = 7.0) -> str:
    s = RunState(run_id="r", run_dir="runs/r", topic="t", max_rounds=max_rounds, pass_score=pass_score,
                 critic_rounds=list(reports))
    return graph.route_after_critic(s)


def test_weak_scenes_go_back_for_revision():
    assert routed(report(5, [3])) == "revise"


def test_a_passing_score_ends_even_with_a_minor_weak_scene():
    assert routed(report(7.5, [3])) == graph.END


def test_nothing_flagged_ends_even_with_a_low_score():
    assert routed(report(4, [])) == graph.END


def test_max_rounds_ends_the_loop():
    assert routed(report(5, [3]), report(5, [3], 1)) == "revise"  # one revision done, one left
    assert routed(report(5, [3]), report(5, [3], 1), report(5, [3], 2)) == graph.END
    assert routed(report(5, [3]), max_rounds=0) == graph.END


# --- reducers ------------------------------------------------------------------------------------------


def test_usage_is_summed_per_node():
    merged = add_usage({"critic": Usage(calls=1, prompt_tokens=100, cost_usd=0.1, seconds=2)},
                       {"critic": Usage(calls=1, prompt_tokens=50, cost_usd=0.05, seconds=1),
                        "writer": Usage(calls=2)})
    critic_usage = merged["critic"]
    assert (critic_usage.calls, critic_usage.prompt_tokens, critic_usage.seconds) == (2, 150, 3)
    assert critic_usage.cost_usd == pytest.approx(0.15)
    assert merged["writer"].calls == 2


def test_the_state_without_reducers_would_lose_history():
    graph_state = graph.build().compile()
    channels = graph_state.channels
    assert type(channels["critic_rounds"]).__name__ == "BinaryOperatorAggregate"  # appended
    assert type(channels["usage"]).__name__ == "BinaryOperatorAggregate"  # summed
    assert type(channels["plan"]).__name__ == "LastValue"  # replaced


# --- checkpoints: crash and resume ----------------------------------------------------------------------


class Killed(Exception):
    pass


def test_a_crashed_run_resumes_without_paying_again(world, monkeypatch):
    def kill(code):
        raise Killed(code)

    monkeypatch.setattr(graph, "hard_exit", kill)
    monkeypatch.setenv(graph.CRASH_ENV, "director")
    state = graph.new_state("coffee shop in Tashkent")
    with graph.open_graph(world.db) as app, pytest.raises(Killed):
        graph.drive(app, state.model_dump(), state.run_id, approve=lambda q: "yes")
    assert world.llm.calls == {"ScriptOptions": 1, "WriterPick": 1}  # the writer finished; the director never started
    paid_before = ledger.entries(Path("runs/ledger.jsonl"), state.run_id)

    monkeypatch.delenv(graph.CRASH_ENV)
    with graph.open_graph(world.db) as app:  # a new process would open the same file
        assert app.get_state(graph.config(state.run_id)).next == ("director",)
        values = graph.drive(app, None, state.run_id, approve=lambda q: "yes")

    assert world.llm.calls["ScriptOptions"] == 1 and world.llm.calls["WriterPick"] == 1  # not paid again
    assert Path(values["final_path"]).exists()
    paid_after = ledger.entries(Path("runs/ledger.jsonl"), state.run_id)
    assert paid_after[: len(paid_before)] == paid_before
    assert [e.step for e in paid_after[len(paid_before):]][0] == "director"


def test_every_node_leaves_a_checkpoint(world):
    state, _ = start(world)
    with graph.open_graph(world.db) as app:
        history = list(app.get_state_history(graph.config(state.run_id)))
    nexts = [h.next[0] if h.next else "END" for h in reversed(history)]
    assert nexts[:8] == ["__start__", "writer", "director", "approve_hero", "producer", "editor", "critic", "revise"]
    assert nexts[-1] == "END"


def test_checkpoints_only_unpack_our_own_classes(world):
    state, _ = start(world)
    rows = sqlite3.connect(world.db).execute("select count(*) from checkpoints where thread_id = ?", (state.run_id,))
    assert rows.fetchone()[0] > 10
    assert RunState in graph.SAFE_TYPES and writer.WriterPick in graph.SAFE_TYPES


# --- the approval interrupt --------------------------------------------------------------------------------


def test_yes_makes_the_hero_clip(world):
    _, values = start(world, approve="yes")
    s = RunState.model_validate(values)
    assert s.hero.approved is True and world.made["clips"] == ["scene_01.mp4"]
    assert s.plan.scenes[0].motion == "ai_clip"


def test_no_still_finishes_the_video_with_ken_burns_for_that_scene(world):
    _, values = start(world, approve="no")
    s = RunState.model_validate(values)
    assert s.hero.approved is False and world.made["clips"] == []
    assert s.plan.scenes[0].motion == "pan_left"  # the director's own move for that shot
    assert b"ai_clip" not in Path(s.final_path).read_bytes()  # the editor got no clip scene


def test_without_an_answer_the_run_pauses_and_resumes_later(world):
    state, values = start(world, approve=None)
    with graph.open_graph(world.db) as app:
        snapshot = app.get_state(graph.config(state.run_id))
        assert snapshot.next == ("approve_hero",)
        question = snapshot.interrupts[0].value
        assert "~$0.30" in question["question"] and question["mock"] is True and question["scene"] == 1
        assert world.llm.calls == {"ScriptOptions": 1, "WriterPick": 1, "DirectorPlan": 1}
        values = graph.drive(app, graph.Command(resume="yes"), state.run_id)
    assert Path(values["final_path"]).exists() and world.made["clips"] == ["scene_01.mp4"]
    assert world.llm.calls["DirectorPlan"] == 1  # nothing before the interrupt ran again


def test_a_revision_of_the_hero_image_asks_again(world):
    world.llm.critic_scores = [(5.0, [1]), (8.0, [])]
    questions = []
    state = graph.new_state("coffee shop in Tashkent")
    with graph.open_graph(world.db) as app:
        graph.drive(app, state.model_dump(), state.run_id, approve=lambda q: questions.append(q) or "yes")
    assert len(questions) == 2 and world.made["clips"] == ["scene_01.mp4", "scene_01.mp4"]


# --- tracing ----------------------------------------------------------------------------------------------


class FakeMlflow:
    """Records spans; start_span can be told to fail."""

    def __init__(self, fail: bool = False):
        self.spans, self.fail = [], fail

    def start_span(self, name, span_type):
        if self.fail:
            raise RuntimeError("the tracking store is locked")
        record = {"name": name, "type": span_type}
        self.spans.append(record)
        live = NS(set_inputs=lambda v: record.update(inputs=v), set_outputs=lambda v: record.update(outputs=v),
                  set_attributes=lambda v: record.update(attributes=v), trace_id="tr-1")

        class Manager:
            def __enter__(self):
                return live

            def __exit__(self, *exc):
                record["error"] = exc[0] is not None
                return False

        return Manager()

    def update_current_trace(self, tags):
        pass


def test_every_node_gets_a_span_with_tokens_and_seconds(world, monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setattr(tracing, "_mlflow", fake)
    start(world)
    names = [s["name"] for s in fake.spans]
    assert names[:7] == ["writer", "director", "approve_hero", "approve_hero", "producer", "editor", "critic"]
    writer_span = fake.spans[0]
    assert writer_span["attributes"]["calls"] == 2 and writer_span["attributes"]["prompt_tokens"] == 200
    assert writer_span["outputs"]["chosen"]["title"] == "Option 2"  # a summary, not the whole script
    assert writer_span["inputs"]["prompt_version"] == "v2"
    paused = fake.spans[2]  # the interrupt: closed normally, marked, no error
    assert paused["attributes"] == {"interrupted": True} and paused["error"] is False


def test_tracing_off_or_broken_never_breaks_a_run(world, monkeypatch):
    monkeypatch.setattr(tracing, "_mlflow", FakeMlflow(fail=True))
    _, values = start(world)
    assert Path(values["final_path"]).exists()
    monkeypatch.setattr(tracing, "_mlflow", None)
    with tracing.span("x") as s:
        s.set_outputs({"a": 1})  # a no-op, no error
    assert s.trace_id is None


def test_tracing_setup_without_mlflow_returns_false(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("No module named 'mlflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlflow)
    monkeypatch.setattr(tracing, "_mlflow", None)
    assert tracing.setup() is False and not tracing.enabled()


# --- the CLI ---------------------------------------------------------------------------------------------------


def test_resume_with_a_topic_is_refused(monkeypatch):
    from shortsmith import make

    monkeypatch.setattr("sys.argv", ["make", "topic", "--resume", "2026-10-23_120000"])
    with pytest.raises(SystemExit):
        make.main()


def test_the_producer_never_offers_the_paid_clip_to_the_agent_loop():
    from shortsmith.agents import tools

    assert [s["function"]["name"] for s in tools.schemas()] == ["generate_image", "text_to_speech", "render_video"]
    assert tools.TOOLS["animate_image"].group == "clip"
    assert producer.fingerprint("a") != producer.fingerprint("b")


# --- 6.8 and 6.9 tools, offline -----------------------------------------------------------------------------


def test_context_eval_rebuilds_each_nodes_request_from_the_checkpoints(world):
    from shortsmith.agents import context_eval

    state, _ = start(world)
    with graph.open_graph(world.db) as app:
        built = context_eval.requests(app, state.run_id)
    assert list(built) == ["writer", "writer-pick", "director", "critic", "revise"]
    sent = {r["response_format"]["json_schema"]["name"]: r["messages"] for r in world.llm.requests}
    assert built["director"][0] == sent["DirectorPlan"]  # rebuilt byte for byte from the checkpointed state
    assert built["writer"][0] == sent["ScriptOptions"]
    messages, _, _, s = built["director"]
    full = context_eval.with_everything(messages, s)
    trimmed_text, full_text = context_eval.text_of(messages), context_eval.text_of(full)
    assert full_text.startswith(trimmed_text) and "script_options" in full_text  # the whole state rides along
    assert len(full_text) > 2 * len(trimmed_text)  # here without research; a real run adds research.json too


def test_the_prompt_evaluation_scores_both_versions_on_the_same_notes(world):
    from shortsmith.agents import prompt_eval
    from shortsmith.models import HookIdea, ResearchNotes

    notes = ResearchNotes(trending=[], hooks=[HookIdea(pattern="Stop", hook="Stop scrolling now.", why="w")],
                          angles=[], avoid=[], visual_style="")
    world.llm.critic_scores = [(6.0, [2])]
    v1 = prompt_eval.run_version("coffee", "en", notes, "v1")
    v2 = prompt_eval.run_version("coffee", "en", notes, "v2")
    systems = [r["messages"][0]["content"] for r in world.llm.requests
               if r["response_format"]["json_schema"]["name"] == "ScriptOptions"]
    assert systems == [writer.PROMPTS["v1"], writer.PROMPTS["v2"]]
    critic_texts = [r["messages"][1]["content"][0]["text"] for r in world.llm.requests
                    if r["response_format"]["json_schema"]["name"] == "CriticVerdict"]
    assert all("v1" not in t and "v2" not in t for t in critic_texts)  # the judge is blind to the version
    m = prompt_eval.metrics(v1)
    assert m["critic_score"] == 6.0 and m["options_valid"] == 2 and m["options_rejected"] == 1
    assert m["chosen_hook_copied"] == 1.0 and v1.review.checks.hook_copied_from == "notes: Stop"
    table = prompt_eval.report([prompt_eval.TopicResult(topic="coffee", language="en", notes=notes,
                                                        results={"v1": v1, "v2": v2})])
    assert "v1: mean critic score 6.00" in table and "v2: mean critic score 6.00" in table


def test_the_cli_runs_pauses_and_resumes_a_graph_run(world, monkeypatch, capsys):
    from shortsmith import make

    monkeypatch.setattr(graph, "CHECKPOINTS", world.db)
    monkeypatch.setattr(make.ffmpeg, "probe", lambda path: ffmpeg.VideoInfo(
        width=1080, height=1920, fps=30, frames=900, duration_s=30.0, size_bytes=1, video_codec="h264",
        pix_fmt="yuv420p", audio_codec="aac"))
    monkeypatch.setattr("sys.stdin", NS(isatty=lambda: False))
    monkeypatch.setattr("sys.argv", ["make", "coffee shop", "--graph", "--no-trace"])
    make.main()
    out = capsys.readouterr().out
    assert "no one to ask: the run is saved and paused here." in out and "resume with:" in out
    run_id = out.split("run ", 1)[1].split(" ", 1)[0]

    monkeypatch.setattr("sys.argv", ["make", "--resume", run_id, "--approve", "no", "--no-trace"])
    make.main()
    out = capsys.readouterr().out
    assert f"resuming {run_id} before approve_hero" in out and "answered by --approve: no" in out
    assert "hero clip refused: Ken Burns instead" in out and "stop: passed (8 >= 7)" in out
    assert "ledger lines for this run: 6 (3 new)" in out  # writer x2 + director before the pause; critic, revise, critic after

    monkeypatch.setattr("sys.argv", ["make", "--resume", run_id, "--no-trace"])
    with pytest.raises(SystemExit, match="already finished"):
        make.main()
