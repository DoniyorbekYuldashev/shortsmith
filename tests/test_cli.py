"""The `shortsmith` command (cli.py), offline: Typer's CliRunner drives every command against the same
fakes as test_graph.py (a fake LLM that answers by schema, fake voice, ffmpeg and frames, a real
LangGraph with a real SQLite checkpointer in a temp folder). Nothing here spends anything: media is
mock or faked, and --real runs only with fake media functions and a fake FAL_KEY."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_graph import option, world  # noqa: F401  (world is a fixture; pytest puts tests/ on sys.path)
from typer.testing import CliRunner

from shortsmith import cli, console, ledger, runs
from shortsmith.agents import graph
from shortsmith.agents.state import RunState
from shortsmith.media import ffmpeg
from shortsmith.research import rag, stores

runner = CliRunner()
ANSI = "\x1b["
REAL_STATUS = stores.status
HEALTHY = stores.StoreStatus(backend="pgvector", table="kb_gemini_embedding_2_768",
                             url="postgresql://shortsmith:***@127.0.0.1:5433/shortsmith", docker="engine 29.4.0",
                             docker_running=True, reachable=True, postgres="17.11", pgvector="0.8.7", rows=79)


@pytest.fixture
def env(world, monkeypatch):  # noqa: F811
    """`world` plus what the CLI checks before spending: an LLM key, ffmpeg on PATH, ffprobe's answer,
    and a healthy vector store for doctor (no Docker or database in unit tests)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("SHORTSMITH_BUDGET", raising=False)
    monkeypatch.delenv("SHORTSMITH_LANG", raising=False)
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    monkeypatch.setattr(stores, "status", lambda collection: HEALTHY)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"C:/fake/{name}.exe")
    monkeypatch.setattr(ffmpeg, "probe", lambda path: ffmpeg.VideoInfo(
        width=1080, height=1920, fps=30, frames=900, duration_s=30.0, size_bytes=3_100_000, video_codec="h264",
        pix_fmt="yuv420p", audio_codec="aac"))
    return world


def run(*args: str, input: str | None = None):
    return runner.invoke(cli.app, list(args), input=input)


def newest() -> Path:
    return runs.run_folders()[0]


def state_of(run_id: str) -> tuple[RunState, tuple]:
    with graph.open_graph() as app:
        snapshot = app.get_state(graph.config(run_id))
    return RunState.model_validate(snapshot.values), snapshot.next


def history(**costs: float) -> None:
    """Past ledger lines (no run id), so budget estimates are known numbers."""
    models = {"critic": "gemini-3.8-flash"}
    for step, usd in costs.items():
        step = step.replace("_", "-")
        ledger.record(ledger.LedgerEntry(step=step, model=models.get(step, "gemini-3.5-flash-lite"), prompt_tokens=0,
                                         completion_tokens=0, cost_usd=usd, run_id=None), ledger.DEFAULT_PATH)


# --- help ------------------------------------------------------------------------------------------------


def test_help_shows_the_command_tree_exit_codes_and_settings_order(env):
    result = run("--help")
    assert result.exit_code == 0
    for command in ("make", "resume", "rerun-scene", "list", "show", "cost", "doctor", "serve", "mcp"):
        assert command in result.stdout
    assert "Exit codes: 0 ok, 1 error, 2 usage, 3 paused for an approval, 4 over budget" in " ".join(result.stdout.split())
    assert "flags > environment variables > .env in the current folder > defaults" in " ".join(result.stdout.split())


@pytest.mark.parametrize("command, options", [
    ("make", ["--lang", "--research", "--brand", "--approve", "--hero-clips", "--budget", "--mock", "--real", "--yes",
              "--max-rounds", "--pass-score", "--critic-model", "--prompt", "--mcp", "--no-trace", "--simple"]),
    ("resume", ["--approve", "--budget", "--mock", "--real", "--yes"]),
    ("rerun-scene", ["--prompt", "--narration", "--image", "--voice", "--approve", "--real"]),
    ("list", ["--limit", "--json"]), ("show", ["--json"]), ("cost", ["--days", "--run", "--json"]),
    ("doctor", []), ("serve", ["--port", "--no-trace"]), ("mcp", ["--root"]),
])
def test_every_command_explains_itself(env, command, options):
    result = run(command, "--help")
    assert result.exit_code == 0
    for name in options:
        assert name in result.stdout
    assert "--crash-before" not in result.stdout  # a demo switch, hidden


def test_bare_shortsmith_shows_the_help(env):
    result = run()
    assert "make" in result.output and "Usage" in result.output


# --- make / resume ---------------------------------------------------------------------------------------------


def test_make_runs_the_graph_to_a_final_video(env):
    result = run("make", "coffee shop in Tashkent", "--approve", "yes", "--no-trace")
    assert result.exit_code == 0, result.output
    folder = newest()
    assert (folder / "final.mp4").exists()
    out, err = result.stdout, result.stderr
    assert "video" in out and str((folder / "final.mp4").resolve()) in out  # results on stdout
    assert "critic   5 -> 8 | passed (8 >= 7)" in out
    assert "cost     $0.0009 list price | $0 billed (free tier / free daily tokens) | 6 calls (6 in this session)" in out
    assert "done writer" in err and "... writer: writing 3 script options" in err  # progress on stderr
    assert "hero clip? Scene 1" in err and "answered by --approve: yes" in err
    assert "| node " in out and "| total" not in out  # a Markdown table when not a terminal
    assert ANSI not in out and ANSI not in err
    assert not (folder / runs.RUNNING).exists()  # the running marker is gone once the process is done


def test_without_an_answer_the_run_pauses_with_exit_3_and_resumes(env):
    result = run("make", "coffee", "--no-trace")
    assert result.exit_code == cli.ExitCode.PAUSED == 3
    run_id = newest().name
    assert "nobody to ask here (no terminal)" in result.stderr
    assert f"shortsmith resume {run_id} --approve yes" in result.stdout
    assert runs.list_runs()[0].status.label == "paused at approval"

    result = run("resume", run_id, "--approve", "no", "--no-trace")
    assert result.exit_code == 0, result.output
    assert f"resuming {run_id} before approve_hero" in result.stderr
    assert "hero clip refused: Ken Burns instead" in result.stderr
    assert env.made["clips"] == []

    again = run("resume", run_id)
    assert again.exit_code == 0 and "already finished" in again.stderr


def test_budget_stops_before_the_step_that_would_pass_it_and_resume_finishes(env):
    history(writer=0.001, writer_pick=0.001, director=0.001, critic=0.01, revise=0.001)
    result = run("make", "coffee", "--budget", "0.004", "--approve", "yes", "--no-trace")
    assert result.exit_code == cli.ExitCode.OVER_BUDGET == 4, result.output
    run_id = newest().name
    s, next_nodes = state_of(run_id)
    assert next_nodes == ("critic",) and s.budget_usd == 0.004  # stopped before the critic, everything before it saved
    assert "over budget before critic" in result.stdout and "nothing was sent for that step" in result.stdout
    assert "the free tier bills $0" in result.stdout
    # spent + every round still allowed (3 critic + 2 revise estimates), rounded up to a cent: a budget that surely finishes
    assert f"shortsmith resume {run_id} --budget 0.04" in result.stdout
    assert [e.step for e in ledger.entries(ledger.DEFAULT_PATH, run_id)] == ["writer", "writer-pick", "director"]
    assert runs.list_runs()[0].status.label == "over budget before critic"

    result = run("resume", run_id, "--budget", "0.05", "--no-trace")
    assert result.exit_code == 0, result.output
    s, next_nodes = state_of(run_id)
    assert next_nodes == () and s.budget_usd == 0.05  # the new budget is saved with the run
    assert env.llm.calls["ScriptOptions"] == 1  # the writer was not paid for again
    assert [e.step for e in ledger.entries(ledger.DEFAULT_PATH, run_id)][3:] == ["critic", "revise", "critic"]


def test_the_budget_in_the_environment_counts_unless_a_flag_says_otherwise(env, monkeypatch):
    monkeypatch.setenv("SHORTSMITH_BUDGET", "0.0001")  # below the writer's first-call guess
    assert run("make", "coffee", "--approve", "yes", "--no-trace").exit_code == 4
    assert env.llm.calls == {}  # stopped before the first call
    assert run("make", "coffee", "--approve", "yes", "--no-trace", "--budget", "1").exit_code == 0


def test_hero_clips_0_means_ken_burns_and_no_question(env):
    result = run("make", "coffee", "--hero-clips", "0", "--no-trace")
    assert result.exit_code == 0, result.output  # never paused: there is nothing to approve
    s, _ = state_of(newest().name)
    assert s.hero is None and "ai_clip" not in [scene.motion for scene in s.plan.scenes]
    assert env.made["clips"] == [] and "hero clip?" not in result.stderr
    assert "no hero clip (--hero-clips 0)" in result.stderr


def test_more_than_one_hero_clip_is_refused_with_the_reason(env):
    result = run("make", "coffee", "--hero-clips", "2")
    assert result.exit_code == 2
    assert "0 or 1 for now" in result.stderr and env.llm.calls == {}


def test_real_without_yes_and_nobody_to_ask_is_refused_before_anything_starts(env, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")  # images: OpenAI's gpt-image-2 at low quality, the default
    result = run("make", "coffee", "--real")
    assert result.exit_code == cli.ExitCode.USAGE
    assert "images: 6 x $0.0050 = $0.0300 (openai gpt-image-2 (low, 1024x1536)" in result.stderr
    assert "hero clip: ~$0.30" in result.stderr
    assert "--real needs a yes, and there is no terminal to ask" in result.stderr and "add --yes" in result.stderr
    assert env.llm.calls == {} and not Path("runs").exists()  # no run folder, no call


def test_real_asks_and_no_spends_nothing(env, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setattr(console, "stdin_interactive", lambda: True)
    result = run("make", "coffee", "--real", input="n\n")
    assert result.exit_code == 1 and "cancelled: nothing was made or spent" in result.stderr
    assert env.llm.calls == {} and not Path("runs").exists()


def test_real_with_yes_runs_with_real_media_recorded_on_the_run(env, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fake")  # every media function is faked in `world`: nothing is sent anywhere
    monkeypatch.setenv("IMAGE_PROVIDER", "fal")
    monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")
    result = run("make", "coffee", "--real", "--yes", "--approve", "no", "--no-trace")
    assert result.exit_code == 0, result.output
    assert "confirmed by --yes" in result.stderr and "media: REAL images, fal fal-ai/flux/schnell" in result.stderr
    s, _ = state_of(newest().name)
    assert s.media == "real"
    assert "... producer: making the images and voices that are missing (est. $0.0180)" in result.stderr


def test_a_mock_flag_wins_over_mock_media_0_in_the_environment(env, monkeypatch):
    monkeypatch.setenv("MOCK_MEDIA", "0")
    result = run("make", "coffee", "--mock", "--approve", "yes", "--no-trace")
    assert result.exit_code == 0 and "--real" not in result.stderr
    assert state_of(newest().name)[0].media == "mock"


def test_a_missing_key_fails_before_anything_is_spent(env, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    result = run("make", "coffee")
    assert result.exit_code == 1
    assert "error: GEMINI_API_KEY is not set (LLM_PROVIDER=gemini)" in result.stderr and "next:" in result.stderr
    assert not Path("runs").exists()


def test_a_failing_node_is_one_message_and_debug_shows_the_traceback(env, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("the director tripped")

    monkeypatch.setattr(graph.director, "direct", boom)
    result = run("make", "coffee", "--no-trace")
    run_id = newest().name
    assert result.exit_code == 1
    assert "error: RuntimeError: the director tripped" in result.stderr
    assert f"shortsmith resume {run_id}" in result.stderr and "Traceback" not in result.output
    assert runs.list_runs()[0].status.label == "crashed at director"

    result = run("--debug", "make", "coffee", "--no-trace")
    assert isinstance(result.exception, RuntimeError)  # --debug lets it through (the runner catches it here)


def test_a_run_another_live_process_drives_shows_as_running_and_cant_be_resumed(env):
    assert run("make", "coffee", "--no-trace").exit_code == 3  # paused: stands in for a run in progress
    folder = newest()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (folder / runs.RUNNING).write_text(json.dumps({"pid": other.pid}), encoding="utf-8")
        assert runs.list_runs()[0].status.label == "running (approve_hero)"
        result = run("resume", folder.name, "--approve", "yes", "--no-trace")
        assert result.exit_code == 1 and f"is running right now (process {other.pid})" in result.stderr
    finally:
        other.kill()
        other.wait()
    assert runs.list_runs()[0].status.label == "paused at approval"  # its process is gone: the checkpoint decides


def test_ctrl_c_and_a_used_up_daily_quota_are_explained():
    from shortsmith import llm

    message, hint, code = cli.explain(KeyboardInterrupt(), "2026-10-24_184531")
    assert (message, code) == ("interrupted", 130) and "shortsmith resume 2026-10-24_184531" in hint
    quota = llm.DailyQuotaError("the free tier's daily request quota for this model is used up (it resets at midnight "
                                "Pacific time); try another model: 429 ... PerDay")
    message, hint, code = cli.explain(quota, "r1")
    assert code == 1 and message.endswith("midnight Pacific time)") and "--critic-model gemini-3.6-flash" in hint


def test_global_flags_may_come_after_the_command():
    assert cli._move_global_flags(["make", "x", "--debug"]) == ["--debug", "make", "x"]
    assert cli._move_global_flags(["show", "--plain", "latest"]) == ["--plain", "show", "latest"]
    assert cli._move_global_flags(["make", "--", "--debug"]) == ["make", "--", "--debug"]  # after --: a value


def test_simple_runs_the_straight_pipeline_and_tags_the_ledger(env):
    original = env.llm.answer
    env.llm.answer = lambda name, request: option(1) if name == "Script" else original(name, request)
    result = run("make", "coffee", "--simple")
    assert result.exit_code == 0, result.output
    folder = newest()
    assert (folder / "final.mp4").exists() and not (folder / "media.json").exists()
    assert [e.step for e in ledger.entries(ledger.DEFAULT_PATH, folder.name)] == ["writer"]
    assert runs.list_runs()[0].status.label == "no checkpoint" and runs.list_runs()[0].cost_usd > 0
    assert run("make", "coffee", "--simple", "--max-rounds", "1").exit_code == 2  # a graph-only flag


# --- list / show / cost ---------------------------------------------------------------------------------------


@pytest.fixture
def three_runs(env):
    """A finished graph run, a run stopped by its budget, and an old folder with no checkpoint."""
    old = Path("runs/2026-10-01_120000")
    old.mkdir(parents=True)
    (old / "script.json").write_text(json.dumps(option(1) | {"title": "An old run"}), encoding="utf-8")
    assert run("make", "coffee", "--approve", "yes", "--no-trace").exit_code == 0
    history(critic=0.5)
    assert run("make", "plov", "--budget", "0.1", "--approve", "yes", "--no-trace").exit_code == 4
    return [f.name for f in runs.run_folders()]


def test_list_shows_runs_newest_first_with_their_status(three_runs):
    result = run("list")
    assert result.exit_code == 0 and ANSI not in result.output
    lines = [line for line in result.stdout.splitlines() if line.startswith("| 2026")]
    assert [line.split("|")[1].strip() for line in lines] == three_runs
    assert "over budget before critic" in lines[0] and "finished" in lines[1] and "no checkpoint" in lines[2]
    assert "Option 2" in lines[1] and re.search(r"\|\s+8 \|", lines[1]) and "An old run" in lines[2]


def test_list_json_is_clean_json_on_stdout(three_runs):
    result = run("list", "--json")
    rows = json.loads(result.stdout)  # nothing else on stdout
    assert [r["id"] for r in rows] == three_runs
    finished = rows[1]
    assert finished["status"] == {"kind": "finished", "node": None, "detail": "", "label": "finished"}
    assert finished["score"] == 8.0 and finished["reviews"] == 2 and finished["language"] == "en"
    assert finished["cost_usd"] == pytest.approx(6 * 0.000155) and finished["billed_usd"] == 0
    assert finished["length_s"] > 0 and finished["final_mp4"].endswith("final.mp4")
    assert rows[0]["status"]["kind"] == "over_budget" and rows[2]["cost_usd"] is None


def test_show_prints_everything_about_one_run(three_runs):
    result = run("show", three_runs[1])
    assert result.exit_code == 0 and ANSI not in result.output
    out = result.stdout
    for part in ("Script", "Critic", "Revisions", "Cost per node", "video", "hero", "history/round_1"):
        assert part in out
    assert "scene 3 off topic" in out and "revised shot 3" in out


def test_show_json_has_the_scenes_rounds_revisions_and_nodes(three_runs):
    d = json.loads(run("show", three_runs[1], "--json").stdout)
    assert len(d["scenes"]) == 6 and [r["score"] for r in d["critic_rounds"]] == [5.0, 8.0]
    assert d["revisions"][0]["scene"] == 3 and d["hero"]["approved"] is True
    assert {n["node"]: n["calls"] for n in d["nodes"]} == {"writer": 2, "director": 1, "approve_hero": 0, "producer": 0,
                                                           "editor": 0, "critic": 2, "revise": 1}
    assert d["settings"]["media"] == "mock" and d["files"]["images"] == 6


def test_show_latest_and_a_unique_prefix_work(three_runs):
    assert json.loads(run("show", "--json").stdout)["id"] == three_runs[0]  # the default is "latest"
    assert json.loads(run("show", "2026-10-01", "--json").stdout)["id"] == three_runs[2]


def test_an_unknown_run_is_one_clear_error(env):
    result = run("show", "no-such-run")
    assert result.exit_code == 1
    assert "error: no run 'no-such-run' in runs/" in result.stderr and "next: `shortsmith list`" in result.stderr
    assert "Traceback" not in result.output


def test_cost_tables_and_json(three_runs):
    result = run("cost")
    assert result.exit_code == 0 and ANSI not in result.output
    for part in ("By day", "By run", "By model and step", "Gemini today", "YouTube", "billed"):
        assert part in result.stdout
    report = json.loads(run("cost", "--json").stdout)
    assert report["total"]["calls"] == len(ledger.entries(ledger.DEFAULT_PATH))
    assert report["total"]["billed_usd"] == 0 and report["free_tier"] is True
    assert {r["key"] for r in report["by_run"]} >= {three_runs[0], three_runs[1], "(no run id)"}
    flash = {m["model"]: m for m in report["quotas"]["gemini"]}
    assert flash["gemini-3.8-flash"]["daily_limit"] == 20
    one = json.loads(run("cost", "--run", three_runs[1], "--json").stdout)
    assert one["total"]["calls"] == 6 and one["run"] == three_runs[1]


# --- rerun-scene ---------------------------------------------------------------------------------------------------


def finished_run(env) -> Path:
    assert run("make", "coffee", "--approve", "yes", "--no-trace").exit_code == 0
    env.made["images"].clear()
    env.made["clips"].clear()
    return newest()


def test_rerun_scene_remakes_that_scene_only_and_keeps_the_old_files(env):
    folder = finished_run(env)
    before = json.loads((folder / "script.json").read_text(encoding="utf-8"))
    lines = len(ledger.entries(ledger.DEFAULT_PATH))
    old_voice = (folder / "audio/scene_03.mp3").stat().st_mtime_ns

    result = run("rerun-scene", folder.name, "3", "--prompt", "a wide shot of the bazaar at dawn")
    assert result.exit_code == 0, result.output
    assert env.made["images"] == ["scene_03.png"] and env.made["clips"] == []  # nothing else made
    assert (folder / "history/rerun_1/scenes/scene_03.png").exists()  # the replaced picture
    assert (folder / "history/rerun_1/final.mp4").exists()  # and the replaced video
    after = json.loads((folder / "script.json").read_text(encoding="utf-8"))
    assert after["scenes"][2]["visual_prompt"] == "a wide shot of the bazaar at dawn, cinematic, warm light"  # style kept
    assert [s for i, s in enumerate(after["scenes"]) if i != 2] == [s for i, s in enumerate(before["scenes"]) if i != 2]
    assert (folder / "audio/scene_03.mp3").stat().st_mtime_ns == old_voice  # the voice was kept
    assert len(ledger.entries(ledger.DEFAULT_PATH)) == lines  # no LLM call, no paid media
    assert "made     images 3 (scene 3; every other scene untouched)" in result.stdout
    assert runs.load_records(folder)[0].label == "rerun_1"
    assert "Reruns" in run("show", folder.name).stdout


def test_rerun_new_narration_remakes_only_the_voice(env):
    folder = finished_run(env)
    result = run("rerun-scene", folder.name, "4", "--narration", "A brand new line for scene four, said out loud.")
    assert result.exit_code == 0, result.output
    assert env.made["images"] == [] and "voices 4" in result.stdout
    assert (folder / "history/rerun_1/audio/scene_04.mp3").exists()


def test_rerun_of_the_hero_picture_asks_first_and_pauses_without_an_answer(env):
    folder = finished_run(env)
    before = (folder / "script.json").read_text(encoding="utf-8")
    result = run("rerun-scene", folder.name, "1", "--prompt", "a new hook picture")
    assert result.exit_code == 3 and "hero clip? Scene 1 is the hero scene" in result.stderr
    assert (folder / "script.json").read_text(encoding="utf-8") == before and not (folder / "history/rerun_1").exists()

    result = run("rerun-scene", folder.name, "1", "--prompt", "a new hook picture", "--approve", "no")
    assert result.exit_code == 0, result.output
    assert "ai_clip -> pan_left" in result.stdout and env.made["clips"] == []
    assert (folder / "history/rerun_1/clips_ai/scene_01.mp4").exists()  # the clip no longer belongs to the scene


def test_rerun_with_yes_makes_a_new_hero_clip(env):
    folder = finished_run(env)
    result = run("rerun-scene", folder.name, "1", "--prompt", "a new hook picture", "--approve", "yes")
    assert result.exit_code == 0, result.output
    assert env.made["clips"] == ["scene_01.mp4"] and "clip 1" in result.stdout


def test_rerun_refuses_an_unfinished_run_and_a_missing_scene(env):
    assert run("make", "coffee", "--no-trace").exit_code == 3  # paused at the approval question
    result = run("rerun-scene", "latest", "2", "--image")
    assert result.exit_code == 1 and "is not finished (paused at approval)" in result.stderr
    run("resume", "latest", "--approve", "no", "--no-trace")
    result = run("rerun-scene", "latest", "9", "--image")
    assert result.exit_code == 1 and "no scene 9: this run has scenes 1-6" in result.stderr


def test_rerun_on_a_straight_run_adopts_its_files_first(env):
    original = env.llm.answer
    env.llm.answer = lambda name, request: option(1) if name == "Script" else original(name, request)
    assert run("make", "coffee", "--simple").exit_code == 0
    folder = newest()
    env.made["images"].clear()
    result = run("rerun-scene", folder.name, "2", "--voice")
    assert result.exit_code == 0, result.output
    assert env.made["images"] == []  # its images count as made from its script: not remade
    assert (folder / "media.json").exists() and (folder / "history/rerun_1/audio/scene_02.mp3").exists()


# --- doctor, plain output, the old module path ----------------------------------------------------------------------


def test_doctor_says_whether_keys_are_set_and_never_their_values(env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-123")
    monkeypatch.delenv("FAL_KEY", raising=False)
    result = run("doctor")
    assert result.exit_code == 0, result.output
    assert "secret-value-123" not in result.output
    rows = {line.split("|")[1].strip(): line for line in result.stdout.splitlines() if line.startswith("| ")}
    assert "| ok " in rows["GEMINI_API_KEY"] and "| set;" in rows["GEMINI_API_KEY"] and "not set;" in rows["FAL_KEY"]
    monkeypatch.delenv("GEMINI_API_KEY")
    result = run("doctor")
    assert result.exit_code == 1 and "not ready: GEMINI_API_KEY" in result.stderr


class FakeConnection:
    """Answers the questions stores.status() asks a database."""

    ANSWERS = {"SHOW server_version": "17.11", "extversion": "0.8.7", "to_regclass": "kb_gemini_embedding_2_768",
               "count(*)": 79}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else query.as_string(None)
        value = next(v for key, v in self.ANSWERS.items() if key in text)
        return type("Cursor", (), {"fetchone": lambda self: (value,)})()


def doctor_rows(result) -> dict[str, str]:
    return {line.split("|")[1].strip(): line for line in result.stdout.splitlines() if line.startswith("| ")}


def test_doctor_shows_the_vector_store_and_never_the_database_password(env, monkeypatch):
    monkeypatch.setattr(stores, "status", REAL_STATUS)
    monkeypatch.setenv("DATABASE_URL", "postgresql://shortsmith:pw-123-secret@127.0.0.1:5433/shortsmith")
    monkeypatch.setattr(stores, "connect", lambda url, vectors=True: FakeConnection())
    monkeypatch.setattr(stores, "docker_engine", lambda: (True, "engine 29.4.0"))
    result = run("doctor")
    assert result.exit_code == 0, result.output
    assert "pw-123-secret" not in result.output
    rows = doctor_rows(result)
    assert "pgvector (the default): table kb_gemini_embedding_2_768 at postgresql://shortsmith:***@127.0.0.1:5433/shortsmith" \
        in rows["vector store"]
    assert "| ok " in rows["Docker"] and "engine 29.4.0" in rows["Docker"]
    assert "| ok " in rows["database"] and "PostgreSQL 17.11, pgvector 0.8.7; 79 rows in kb_gemini_embedding_2_768" in rows["database"]


def test_doctor_fails_when_the_selected_database_is_down(env, monkeypatch):
    monkeypatch.setattr(stores, "status", REAL_STATUS)
    monkeypatch.setenv("DATABASE_URL", "postgresql://shortsmith:pw@127.0.0.1:5433/shortsmith")

    def down(url, vectors=True):
        raise stores.StoreUnavailable("can't connect to the pgvector database at 127.0.0.1:5433/shortsmith "
                                      "(connection timeout expired)")

    monkeypatch.setattr(stores, "connect", down)
    monkeypatch.setattr(stores, "docker_engine", lambda: (False, "the engine is not running (start Docker Desktop)"))
    result = run("doctor")
    assert result.exit_code == 1 and "not ready: database" in result.stderr
    rows, text = doctor_rows(result), " ".join(result.stdout.split())
    assert "| FAIL " in rows["database"] and "can't connect to the pgvector database" in rows["database"]
    assert "docker compose up -d db" in text and "VECTOR_STORE=chroma" in text
    assert "| -  " in rows["Docker"] and "start Docker Desktop" in rows["Docker"]  # information: a database can live elsewhere


def test_doctor_with_chroma_selected_needs_no_database(env, monkeypatch):
    monkeypatch.setattr(stores, "status", REAL_STATUS)
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setattr(stores, "connect", lambda *a, **k: pytest.fail("connected to a database"))
    result = run("doctor")
    assert result.exit_code == 0, result.output
    rows = doctor_rows(result)
    assert "chroma (VECTOR_STORE=chroma): collection kb-gemini-embedding-2-768" in rows["vector store"]
    assert "database" not in rows and "Docker" not in rows


def test_doctor_names_an_unknown_vector_store(env, monkeypatch):
    monkeypatch.setattr(stores, "status", REAL_STATUS)
    monkeypatch.setenv("VECTOR_STORE", "pinecone")
    result = run("doctor")
    assert result.exit_code == 1 and "| FAIL " in doctor_rows(result)["vector store"]
    assert "VECTOR_STORE='pinecone'; it must be one of: pgvector, chroma" in result.stdout


def test_research_with_the_database_down_stops_before_any_run_or_call(env, monkeypatch):
    """No quiet fallback to Chroma: one error, the next step, exit 1, and nothing made or spent."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rag.default_kb.cache_clear()
    result = run("make", "coffee shop in Tashkent", "--research", "--approve", "yes", "--no-trace")
    assert result.exit_code == 1, result.output
    assert "error: VECTOR_STORE=pgvector, but DATABASE_URL is not set" in result.stderr
    assert "docker compose up -d db" in result.stderr and "VECTOR_STORE=chroma" in result.stderr
    assert runs.run_folders() == [] and env.llm.calls == {} and ledger.entries(ledger.DEFAULT_PATH) == []
    assert not Path("chroma").exists()


def test_a_store_lost_mid_run_says_how_to_go_on():
    message, hint, code = cli.explain(stores.StoreUnavailable("lost the connection to the pgvector database"), "2026-10-25_101500")
    assert message == "lost the connection to the pgvector database" and code == cli.ExitCode.ERROR
    assert hint == f"{stores.NEXT_STEP}; then shortsmith resume 2026-10-25_101500"


def test_no_colour_codes_even_with_no_color_and_plain(env, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert run("make", "coffee", "--approve", "yes", "--no-trace").exit_code == 0
    for args in (["list"], ["show"], ["cost"], ["--plain", "list"]):
        assert ANSI not in run(*args).output


def test_output_writes_utf8_itself_so_powershell_cant_garble_it(env):
    folder = finished_run(env)
    words = "Qadimgi hovlida tong salqini — oʻzgacha sokinlik ☕, bir piyola choy bilan."
    assert run("rerun-scene", folder.name, "2", "--narration", words).exit_code == 0
    result = run("show", folder.name, "-o", "show.md")
    assert result.exit_code == 0 and result.stdout == "" and "wrote" in result.stderr
    text = Path("show.md").read_bytes().decode("utf-8")  # strict UTF-8, no BOM
    assert words in text and ANSI not in text and not text.startswith("﻿")
    assert run("list", "--json", "-o", "runs.json").exit_code == 0
    assert json.loads(Path("runs.json").read_text(encoding="utf-8"))[0]["id"] == folder.name
    data = json.loads(run("show", folder.name, "--json").stdout)
    assert data["scenes"][1]["narration"] == words and "\\u02bb" in run("show", folder.name, "--json").stdout  # ASCII-escaped


def test_python_m_shortsmith_make_and_python_m_shortsmith_still_work():
    old = subprocess.run([sys.executable, "-m", "shortsmith.make", "--help"], capture_output=True, text=True, timeout=120)
    assert old.returncode == 0 and "--graph" in old.stdout and "--resume" in old.stdout
    new = subprocess.run([sys.executable, "-m", "shortsmith", "--help"], capture_output=True, text=True, timeout=120,
                         encoding="utf-8")
    assert new.returncode == 0 and "rerun-scene" in new.stdout and "Exit codes" in new.stdout


# --- providers per role (the OpenAI switch, 2026-10-26) ------------------------------------------------------------


def test_doctor_shows_each_roles_provider_and_never_a_key(env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SECRET-openai-1")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-SECRET-gemini-2")
    monkeypatch.setenv("OPENAI_DATA_SHARING", "1")
    result = run("doctor")
    assert result.exit_code == 0, result.output
    assert "SECRET" not in result.output
    rows = doctor_rows(result)
    assert "openai: gpt-5-mini (researcher, writer, director, revise; reasoning effort low)" in rows["text"]
    assert "openai: gpt-5.1" in rows["critic"] and "| ok " in rows["critic"]
    assert "gemini: gemini-embedding-2 (768 numbers a vector; the RAG index, Gemini whatever LLM_PROVIDER says)"         in rows["embeddings"]
    assert "openai: gpt-image-2 (low, 1024x1536, ~$0.0050 an image; now: placeholders" in rows["images (--real)"]
    assert "| ok " in rows["OPENAI_API_KEY"] and "| ok " in rows["GEMINI_API_KEY"]  # both needed: text and the index
    assert "data sharing on; 2,500,000 mini + 250,000 large a day; past them: warn" in rows["free tokens"]
    assert "OPENAI_DATA_SHARING=set, OPENAI_FREE_DAILY_TOKENS_MINI=default" in rows["free tokens"]
    monkeypatch.delenv("OPENAI_API_KEY")
    result = run("doctor")
    assert result.exit_code == 1 and "not ready: OPENAI_API_KEY" in result.stderr


def test_doctor_names_an_unknown_image_provider(env, monkeypatch):
    monkeypatch.setenv("IMAGE_PROVIDER", "dalle")
    result = run("doctor")
    assert result.exit_code == 1 and "IMAGE_PROVIDER='dalle'" in result.stdout


def test_no_credit_and_the_free_allowance_stop_have_their_own_next_step():
    from shortsmith import budget, llm

    message, hint, code = cli.explain(llm.NoCreditError("the OpenAI organization has no credit left (429 "
                                                        "insufficient_quota): add credit in ..."), "r1")
    assert message.endswith("(429 insufficient_quota)") and "Settings -> Billing" in hint and "shortsmith resume r1" in hint
    message, hint, code = cli.explain(budget.FreeAllowanceReached("critic", "large", 240_000, 6_000, 250_000), "r1")
    assert "large group: 240,000 used of 250,000" in message and code == cli.ExitCode.OVER_BUDGET
    assert "00:00 UTC" in hint and hint.endswith("shortsmith resume r1")


def test_real_with_a_hero_clip_needs_fals_key_before_anything_starts(env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.delenv("FAL_KEY", raising=False)
    result = run("make", "coffee", "--real", "--yes")
    assert result.exit_code == 1 and "FAL_KEY is not set, and an approved hero clip is made on fal.ai" in result.stderr
    assert "--hero-clips 0" in result.stderr and not Path("runs").exists()
    monkeypatch.delenv("OPENAI_API_KEY")
    result = run("make", "coffee", "--real", "--yes", "--hero-clips", "0")
    assert result.exit_code == 1 and "OPENAI_API_KEY is not set, and --real makes images with openai" in result.stderr
