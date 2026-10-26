"""The web API (api.py, jobs.py, events.py), offline: FastAPI's TestClient against the same fakes as
test_graph.py (a fake LLM answering by schema, fake voice, ffmpeg and frames; a real LangGraph with a
real SQLite checkpointer in a temp folder). The worker thread is real, so queueing is tested as it runs.
Nothing here spends anything or needs the network, Docker or the database."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.sse import format_sse_event
from fastapi.testclient import TestClient
from test_graph import world  # noqa: F401  (a fixture; pytest puts tests/ on sys.path)

from shortsmith import api, cli, events, jobs, ledger, runs
from shortsmith.agents import critic, graph, writer
from shortsmith.agents.state import RunState
from shortsmith.media import ffmpeg
from shortsmith.research import rag, stores

REPO = Path(__file__).resolve().parents[1]
REAL_STATUS = stores.status  # before `env` replaces it with HEALTHY
HEALTHY =stores.StoreStatus(backend="pgvector", table="kb_gemini_embedding_2_768",
                             url="postgresql://shortsmith:***@127.0.0.1:5433/shortsmith", docker="engine 29.4.0",
                             docker_running=True, reachable=True, postgres="17.11", pgvector="0.8.7", rows=79)


@pytest.fixture
def env(world, monkeypatch):  # noqa: F811
    """`world` plus what a run checks before it starts: a key, ffmpeg on PATH, ffprobe, a healthy store."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    monkeypatch.setattr(stores, "status", lambda collection: HEALTHY)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"C:/fake/{name}.exe")
    monkeypatch.setattr(ffmpeg, "probe", lambda path: ffmpeg.VideoInfo(
        width=1080, height=1920, fps=30, frames=900, duration_s=30.0, size_bytes=1, video_codec="h264",
        pix_fmt="yuv420p", audio_codec="aac"))
    monkeypatch.setattr(api, "POLL_S", 0.01)
    return world


@pytest.fixture
def client(env):
    with TestClient(api.create_app(trace=False)) as c:
        yield c


def jobs_of(c: TestClient) -> jobs.Jobs:
    return c.app.state.jobs


def settle(c: TestClient) -> None:
    assert jobs_of(c).worker.wait_idle(30), "the worker didn't finish"


def start(c: TestClient, **body) -> str:
    r = c.post("/runs", json={"topic": "coffee shop in Tashkent", **body})
    assert r.status_code == 202, r.text
    return r.json()["run_id"]


def log_of(c: TestClient, run_id: str) -> list[dict]:
    return c.get(f"/runs/{run_id}/log").json()


def kinds(c: TestClient, run_id: str) -> list[str]:
    return [e["type"] for e in log_of(c, run_id)]


def sse(text: str) -> list[dict]:
    """Parse a text/event-stream body into its messages (comments dropped)."""
    messages = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields: dict = {}
        for line in block.split("\n"):
            if line and not line.startswith(":"):
                key, _, value = line.partition(":")
                fields[key] = fields.get(key, "") + value.removeprefix(" ")
        if "data" in fields:
            messages.append(fields)
    return messages


def history(**costs: float) -> None:
    """Past ledger lines (no run id), so budget estimates are known numbers (as in test_cli)."""
    models = {"critic": "gemini-3.8-flash"}
    for step, usd in costs.items():
        step = step.replace("_", "-")
        ledger.record(ledger.LedgerEntry(step=step, model=models.get(step, "gemini-3.5-flash-lite"), prompt_tokens=0,
                                         completion_tokens=0, cost_usd=usd, run_id=None), ledger.DEFAULT_PATH)


def state_of(run_id: str) -> tuple[RunState, tuple]:
    with graph.open_graph() as app:
        snapshot = app.get_state(graph.config(run_id))
    return RunState.model_validate(snapshot.values), snapshot.next


# --- a run, start to finish ---------------------------------------------------------------------------------


def test_a_run_streams_its_events_in_order_and_ends_with_done(client, env):
    run_id = start(client, approve="yes")
    settle(client)

    with client.stream("GET", f"/runs/{run_id}/events") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        body = r.read().decode()  # a finished run: the stream replays everything and ends after `done`
    assert body.startswith(f": run {run_id}\nretry: 3000\n\n")
    messages = sse(body)
    ids = [int(m["id"]) for m in messages]
    assert ids == list(range(1, len(ids) + 1))  # 1, 2, 3, ... no gaps
    assert all(m["event"] == json.loads(m["data"])["type"] for m in messages)
    order = [json.loads(m["data"])["type"] + (f":{json.loads(m['data']).get('node', '')}" if m["event"].startswith("node") else "")
             for m in messages]
    assert order[:6] == ["queued", "run_started", "node_started:writer", "node_finished:writer", "node_started:director",
                         "node_finished:director"]
    assert order[6:10] == ["node_started:approve_hero", "approval_needed", "approval_answered", "node_started:approve_hero"]
    assert order[-1] == "done" and order.count("critic_report") == 2 and order.count("revision") == 1
    assert order.count("video_made") == 2  # the editor cut it twice: before and after the revision

    data = [json.loads(m["data"]) for m in messages]
    writer_done = next(e for e in data if e["type"] == "node_finished" and e["node"] == "writer")
    assert (writer_done["calls"], writer_done["prompt_tokens"], writer_done["completion_tokens"]) == (2, 200, 100)
    assert writer_done["cost_usd"] > 0 and "picked 2" in writer_done["summary"]
    made = [(e["kind"], e["scene"]) for e in data if e["type"] == "media_made"]
    assert made[:3] == [("image", 1), ("voice", 1), ("image", 2)] and ("clip", 1) in made
    assert ("image", 3) in made[made.index(("clip", 1)) + 1:]  # round 1 remade only the weak scene's picture
    first, second = [e for e in data if e["type"] == "critic_report"]
    assert (first["score"], first["then"], [w["scene"] for w in first["weak_scenes"]]) == (5.0, "revise", [3])
    assert (second["score"], second["then"]) == (8.0, "done") and second["reason"].startswith("passed")
    assert first["frame_urls"][0] == f"/runs/{run_id}/files/critic/round_0/scene_{first['frames'][0]:02d}.jpg?v=0"
    revision = next(e for e in data if e["type"] == "revision")
    assert revision["round"] == 0 and revision["scenes"][0]["scene"] == 3 and revision["scenes"][0]["changed"] is True
    done = data[-1]
    lines = ledger.entries(ledger.DEFAULT_PATH, run_id)
    assert done["calls"] == len(lines) == 6 and done["cost_usd"] == pytest.approx(sum(e.cost_usd for e in lines))
    assert done["billed_usd"] == 0 and done["scores"] == [5.0, 8.0]
    assert done["final_url"] == f"/runs/{run_id}/files/final.mp4?v=1"  # the round of the last cut, not the next one
    started = data[1]
    assert started["free_tier"] is True and started["media"] == "mock" and started["approve"] == "yes"


def test_the_wire_format_is_id_event_data_and_a_blank_line(client):
    run_id = start(client, approve="yes")
    settle(client)
    body = client.get(f"/runs/{run_id}/events").text
    first = body.split("\n\n")[1]  # after the comment + retry block
    assert first.startswith("event: queued\ndata: {") and first.endswith("\nid: 1")
    event = events.EventLog(runs.RUNS).after(run_id)[0]
    assert format_sse_event(data_str=event.model_dump_json(), event="queued", id="1").decode() == first + "\n\n"


def test_without_an_answer_the_run_pauses_and_the_answer_continues_it(client, env):
    run_id = start(client)  # approve="ask": pause at the hero-clip question
    settle(client)
    last = log_of(client, run_id)[-1]
    assert last["type"] == "approval_needed" and last["waiting"] is True and "~$0.30" in last["question"]
    assert client.get(f"/runs/{run_id}").json()["status"]["kind"] == "paused"
    assert env.llm.calls == {"ScriptOptions": 1, "WriterPick": 1, "DirectorPlan": 1}
    assert client.post(f"/runs/{run_id}/resume", json={}).status_code == 409  # a question waits: answer it instead

    r = client.post(f"/runs/{run_id}/approval", json={"answer": "yes"})
    assert r.status_code == 202, r.text
    settle(client)
    order = kinds(client, run_id)
    assert order[order.index("approval_needed") + 1:][:3] == ["approval_answered", "queued", "run_started"]
    assert order[-1] == "done" and env.made["clips"] == ["scene_01.mp4"]
    answered = next(e for e in log_of(client, run_id) if e["type"] == "approval_answered")
    assert answered["by"] == "person" and answered["answer"] == "yes"
    assert env.llm.calls["DirectorPlan"] == 1  # nothing before the question ran again
    assert client.post(f"/runs/{run_id}/approval", json={"answer": "no"}).status_code == 409  # nothing to answer now


def test_no_means_ken_burns_for_the_hero_scene(client, env):
    run_id = start(client)
    settle(client)
    client.post(f"/runs/{run_id}/approval", json={"answer": "no"})
    settle(client)
    s, _ = state_of(run_id)
    assert s.hero.approved is False and env.made["clips"] == [] and s.plan.scenes[0].motion == "pan_left"


def test_a_budget_stop_then_a_resume_with_more_finishes_without_paying_again(client, env):
    history(writer=0.001, writer_pick=0.001, director=0.001, critic=0.01, revise=0.001)
    run_id = start(client, approve="yes", budget_usd=0.004)
    settle(client)
    stop = log_of(client, run_id)[-1]
    assert stop["type"] == "budget_stopped" and stop["node"] == "critic"
    assert stop["limit_usd"] == 0.004 and stop["suggested_usd"] == 0.04  # 3 critic + 2 revise estimates still allowed
    assert client.get("/runs").json()[0]["status"]["label"] == "over budget before critic"

    r = client.post(f"/runs/{run_id}/resume", json={"budget_usd": 0.05})
    assert r.status_code == 202, r.text
    settle(client)
    assert kinds(client, run_id)[-1] == "done"
    s, next_nodes = state_of(run_id)
    assert next_nodes == () and s.budget_usd == 0.05  # saved with the run, as `shortsmith resume --budget`
    assert env.llm.calls["ScriptOptions"] == 1  # the writer was not paid for again
    resumed = [e for e in log_of(client, run_id) if e["type"] == "run_started"][1]
    assert resumed["resumed"] is True and resumed["next"] == ["critic"] and resumed["spent_usd"] > 0
    assert client.post(f"/runs/{run_id}/resume", json={}).status_code == 409  # finished: nothing to do


def test_a_failing_node_is_the_clis_message_and_resume_goes_on(client, env, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("the director tripped")

    real = graph.director.direct
    monkeypatch.setattr(graph.director, "direct", boom)
    run_id = start(client, approve="yes")
    settle(client)
    failed = log_of(client, run_id)[-1]
    assert failed["type"] == "failed" and failed["node"] == "director" and failed["interrupted"] is False
    assert failed["message"] == "RuntimeError: the director tripped" and f"shortsmith resume {run_id}" in failed["hint"]
    assert client.get(f"/runs/{run_id}").json()["status"]["label"] == "crashed at director"

    monkeypatch.setattr(graph.director, "direct", real)
    assert client.post(f"/runs/{run_id}/resume", json={}).status_code == 202
    settle(client)
    assert kinds(client, run_id)[-1] == "done" and env.llm.calls["ScriptOptions"] == 1


# --- replay and reconnect ------------------------------------------------------------------------------------


def test_last_event_id_replays_only_what_came_after(client):
    run_id = start(client, approve="yes")
    settle(client)
    total = len(log_of(client, run_id))
    again = sse(client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "5"}).text)
    assert [int(m["id"]) for m in again] == list(range(6, total + 1))
    assert sse(client.get(f"/runs/{run_id}/events?after=5").text) == again  # the same, for a first connect
    assert [e["id"] for e in client.get(f"/runs/{run_id}/log?after={total - 2}").json()] == [total - 1, total]
    # everything seen and the run is finished: 204 tells EventSource to stop reconnecting
    assert client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(total)}).status_code == 204


def test_a_stream_opened_mid_run_replays_and_then_follows_it_live(client, env, monkeypatch, request):
    gate, review = threading.Event(), critic.review
    request.addfinalizer(gate.set)  # a failing assert must not leave the worker blocked

    def slow_review(*args, **kwargs):
        assert gate.wait(20), "the test never opened the gate"
        return review(*args, **kwargs)

    monkeypatch.setattr(critic, "review", slow_review)
    run_id = start(client, approve="yes")
    seen = []
    with client.stream("GET", f"/runs/{run_id}/events") as r:
        for message in sse_lines(r.iter_lines()):
            seen.append(message["event"])
            if message["event"] == "node_started" and json.loads(message["data"])["node"] == "critic" and not gate.is_set():
                assert client.get(f"/runs/{run_id}").json()["status"] == {
                    "kind": "running", "node": "critic", "detail": "the web server is working on it", "label": "running (critic)"}
                gate.set()  # the critic may answer now: the rest arrives live on the same stream
    assert seen[0] == "queued" and seen[-1] == "done" and seen.count("critic_report") == 2
    settle(client)


def sse_lines(lines) -> list[dict]:
    """Messages from a live stream's lines, as they arrive."""
    fields: dict = {}
    for line in lines:
        if line == "":
            if "data" in fields:
                yield fields
            fields = {}
        elif not line.startswith(":"):
            key, _, value = line.partition(":")
            fields[key] = value.removeprefix(" ")


def test_follow_false_ends_after_the_replay_even_when_the_run_waits(client):
    run_id = start(client)  # pauses at the question: the stream would wait for the answer
    settle(client)
    messages = sse(client.get(f"/runs/{run_id}/events?follow=false").text)
    assert messages[-1]["event"] == "approval_needed"


# --- one run at a time -----------------------------------------------------------------------------------------


def test_one_run_at_a_time_and_the_rest_wait_in_order(env, monkeypatch, request):
    gate, write = threading.Event(), writer.write
    request.addfinalizer(gate.set)

    def slow_write(*args, **kwargs):
        assert gate.wait(20)
        return write(*args, **kwargs)

    monkeypatch.setattr(writer, "write", slow_write)
    with TestClient(api.create_app(trace=False, max_queued=1)) as c:
        first = start(c, approve="yes")
        r = c.post("/runs", json={"topic": "tea in Samarkand", "approve": "yes"})
        second = r.json()["run_id"]
        assert r.json()["status"]["kind"] == "queued" and r.json()["queued"]["ahead"] == [first]
        assert r.json()["queued"]["position"] == 2
        full = c.post("/runs", json={"topic": "a third one"})
        assert full.status_code == 429 and "already waiting" in full.json()["detail"]["message"]
        for _ in range(200):  # the worker reaches the writer's on_start hook on its own time
            if (q := c.get("/queue").json())["node"] == "writer":
                break
            time.sleep(0.05)
        assert q == {"running": first, "node": "writer", "queued": [second], "max_queued": 1}
        listed = {r["id"]: r for r in c.get("/runs").json()}
        assert listed[second]["status"]["kind"] == "queued" and listed[second]["topic"] == "tea in Samarkand"
        assert "run_started" not in kinds(c, second)  # it waits: nothing has started
        gate.set()
        assert c.app.state.jobs.worker.wait_idle(30)
        first_done = log_of(c, first)[-1]
        second_started = next(e for e in log_of(c, second) if e["type"] == "run_started")
        assert first_done["type"] == "done" and second_started["at"] >= first_done["at"]
        assert kinds(c, second)[-1] == "done"


# --- files ------------------------------------------------------------------------------------------------------


def test_files_are_served_from_the_run_folder_only(client):
    run_id = start(client, approve="yes")
    settle(client)
    Path("secret.txt").write_text("TOPSECRET", encoding="utf-8")  # next to runs/, like .env in the repo root
    Path(".env").write_text("GEMINI_API_KEY=TOPSECRET", encoding="utf-8")

    image = client.get(f"/runs/{run_id}/files/scenes/scene_01.png")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert client.get(f"/runs/{run_id}/files/critic/round_0/scene_01.jpg").headers["content-type"] == "image/jpeg"
    for attempt in (f"/runs/{run_id}/files/..%2F..%2Fsecret.txt", f"/runs/{run_id}/files/%2e%2e/%2e%2e/.env",
                    f"/runs/{run_id}/files/..%5C..%5Csecret.txt", f"/runs/{run_id}/files/scenes/..%2F..%2F..%2F.env",
                    f"/runs/{run_id}/files/{Path('secret.txt').resolve().as_posix()}",
                    f"/runs/..%2F..%2F/files/secret.txt", "/runs/.env/files/x.png",
                    f"/runs/{run_id}/files/job.json", f"/runs/{run_id}/files/events.jsonl", f"/runs/{run_id}/files/media.json"):
        r = client.get(attempt)
        assert r.status_code == 404, attempt
        assert "TOPSECRET" not in r.text


def test_the_video_answers_range_requests_so_the_browser_can_seek(client):
    run_id = start(client, approve="yes")
    settle(client)
    whole = client.get(f"/runs/{run_id}/files/final.mp4")
    assert whole.status_code == 200 and whole.headers["accept-ranges"] == "bytes"
    assert whole.headers["content-type"] == "video/mp4" and whole.headers["cache-control"] == "no-cache"
    part = client.get(f"/runs/{run_id}/files/final.mp4", headers={"Range": "bytes=2-5"})
    assert part.status_code == 206 and part.content == whole.content[2:6]
    assert part.headers["content-range"] == f"bytes 2-5/{len(whole.content)}"
    assert client.get(f"/runs/{run_id}/files/final.mp4", headers={"Range": "bytes=999999-"}).status_code == 416


# --- list / show / costs / options ------------------------------------------------------------------------------


def test_list_show_and_costs_are_the_clis_json(client):
    run_id = start(client, approve="yes")
    settle(client)
    rows = client.get("/runs").json()
    assert [r["id"] for r in rows] == [run_id]
    row = rows[0]
    assert row["status"]["label"] == "finished" and row["score"] == 8.0 and row["calls"] == 6 and row["billed_usd"] == 0
    assert runs.RunSummary.model_validate(row) == runs.list_runs()[0]  # the same model `shortsmith list --json` prints

    d = client.get(f"/runs/{run_id}").json()
    assert len(d["scenes"]) == 6 and [r["score"] for r in d["critic_rounds"]] == [5.0, 8.0]
    assert [n["node"] for n in d["nodes"]][:3] == ["writer", "director", "approve_hero"] and d["settings"]["media"] == "mock"
    assert d["files"]["images"] == 6 and d["hero"]["approved"] is True

    report = client.get("/costs").json()
    assert report["total"]["calls"] == 6 and report["by_run"][0]["key"] == run_id and report["free_tier"] is True
    assert client.get(f"/costs?run={run_id}").json()["total"]["calls"] == 6
    missing = client.get("/runs/2026-01-01_000000")
    assert missing.status_code == 404 and missing.json()["detail"] == {"message": "no run '2026-01-01_000000'",
                                                                       "hint": "GET /runs lists them"}
    options = client.get("/options").json()
    assert options["languages"] == ["en", "uz", "ru"] and options["defaults"]["approve"] == "ask"


def test_bad_requests_are_refused_before_anything_starts(client, env):
    assert client.post("/runs", json={"topic": "   "}).status_code == 422
    assert client.post("/runs", json={"topic": "x", "max_rounds": 9}).status_code == 422
    assert client.post("/runs", json={"topic": "x", "hero_clips": 2}).status_code == 422
    assert client.post("/runs", json={"topic": "x", "media": "real"}).status_code == 422  # no --real from the web
    r = client.post("/runs", json={"topic": "x", "brand": "no-such-brand"})
    assert r.status_code == 400 and "no brand" in r.json()["detail"]["message"]
    assert env.llm.calls == {} and not runs.RUNS.exists()


def test_media_is_always_mock_from_the_web(client, env, monkeypatch):
    monkeypatch.setenv("MOCK_MEDIA", "0")  # .env says real: the web still makes placeholders
    history(writer=0.001, writer_pick=0.001, director=0.001, critic=0.01)
    run_id = start(client, approve="yes", budget_usd=0.004)  # stops before the critic: resumable
    settle(client)
    assert state_of(run_id)[0].media == "mock" and api.os.environ["MOCK_MEDIA"] == "1"
    with graph.open_graph() as app:  # as if the CLI had made it with --real
        app.update_state(graph.config(run_id), {"media": "real"})
    r = client.post(f"/runs/{run_id}/resume", json={"budget_usd": 1})
    assert r.status_code == 409 and r.json()["detail"]["hint"] == f"resume it from the CLI: shortsmith resume {run_id}"


def test_a_missing_key_is_refused_with_the_clis_message(client, env, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    r = client.post("/runs", json={"topic": "coffee"})
    assert r.status_code == 503
    assert r.json()["detail"]["message"] == "GEMINI_API_KEY is not set (LLM_PROVIDER=gemini)"
    assert not runs.RUNS.exists()


# --- health, secrets, CORS ---------------------------------------------------------------------------------------


def test_health_with_the_database_down_is_a_clear_status_not_a_crash(client, env, monkeypatch):
    monkeypatch.setattr(stores, "status", REAL_STATUS)
    monkeypatch.setattr(stores, "docker_engine", lambda: (False, "the engine is not running (start Docker Desktop)"))
    monkeypatch.setenv("DATABASE_URL", "postgresql://shortsmith:hunter2pw@127.0.0.1:1/shortsmith")  # nothing listens there
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["ok"] is True and body["research_ready"] is False and body["media"] == "mock"
    assert body["vector_store"]["reachable"] is False and body["vector_store"]["url"].startswith("postgresql://shortsmith:***@")
    assert any("research is off" in p["message"] and "docker compose up -d db" in p["hint"] for p in body["problems"])
    assert "hunter2pw" not in h.text

    rag.default_kb.cache_clear()  # a store another test opened must not answer for the database
    r = client.post("/runs", json={"topic": "coffee", "research": True})
    assert r.status_code == 503 and "docker compose up -d db" in r.json()["detail"]["hint"]
    assert "hunter2pw" not in r.text and not runs.RUNS.exists()  # refused before a run folder exists
    assert env.llm.calls == {}


def test_no_response_holds_a_secret(client, env, monkeypatch):
    secrets = {"GEMINI_API_KEY": "gm-SECRET-key-123", "YOUTUBE_API_KEY": "yt-SECRET-456", "FAL_KEY": "fal-SECRET-789",
               "OPENAI_API_KEY": "sk-SECRET-000"}
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DATABASE_URL", "postgresql://shortsmith:pw-SECRET-db@127.0.0.1:5433/shortsmith")
    run_id = start(client, approve="yes")
    settle(client)
    bodies = [client.get(path).text for path in
              ("/health", "/runs", f"/runs/{run_id}", f"/runs/{run_id}/log", f"/runs/{run_id}/events", "/costs",
               "/options", "/queue", "/openapi.json")]
    assert all("SECRET" not in body for body in bodies)
    health = client.get("/health").json()
    assert health["key_set"] is True and health["youtube_key_set"] is True


def test_cors_answers_only_the_local_frontend(client):
    allowed = client.options("/runs", headers={"Origin": "http://127.0.0.1:3000", "Access-Control-Request-Method": "POST"})
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"
    assert client.get("/queue", headers={"Origin": "http://localhost:3000"}).headers["access-control-allow-origin"] == "http://localhost:3000"
    other = client.get("/queue", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


# --- a restart --------------------------------------------------------------------------------------------------


def test_a_restart_marks_the_running_run_interrupted_and_requeues_the_waiting_one(env, monkeypatch):
    class Killed(Exception):
        pass

    def kill(code):
        raise Killed(code)

    # A run the server was driving when it died: its checkpoint stops before the critic, its job says
    # "running", its last event is the critic starting, and its marker names a process that is gone.
    monkeypatch.setattr(graph, "hard_exit", kill)
    monkeypatch.setenv(graph.CRASH_ENV, "critic")
    state = graph.new_state("coffee shop in Tashkent")
    with graph.open_graph() as app, pytest.raises(Killed):
        graph.drive(app, state.model_dump(), state.run_id, approve=lambda q: "yes")
    monkeypatch.delenv(graph.CRASH_ENV)
    jobs.save_job(jobs.Job(run_id=state.run_id, kind="start", approve="yes", state=state.model_dump(mode="json"),
                           status="running"), runs.RUNS)
    log = events.EventLog(runs.RUNS)
    log.append(state.run_id, events.NodeStarted(node="critic", round=0, doing="judging", estimate_usd=0.0, basis="x"))
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (state.folder / runs.RUNNING).write_text(json.dumps({"pid": gone.pid}), encoding="utf-8")
    # ... and a run that was still waiting in the queue (no checkpoint yet).
    waiting = graph.new_state("tea in Samarkand")
    jobs.save_job(jobs.Job(run_id=waiting.run_id, kind="start", approve="yes", state=waiting.model_dump(mode="json")),
                  runs.RUNS)

    with TestClient(api.create_app(trace=False)) as c:
        assert c.app.state.jobs.worker.wait_idle(30)
        interrupted = log_of(c, state.run_id)[-1]
        assert interrupted["type"] == "failed" and interrupted["interrupted"] is True and interrupted["node"] == "critic"
        assert "the server stopped while this run was at critic" in interrupted["message"]
        assert not (state.folder / runs.RUNNING).exists()
        assert kinds(c, waiting.run_id)[-1] == "done"  # queued again on start, and finished
        assert c.get(f"/runs/{state.run_id}").json()["status"]["label"] == "crashed at critic"
        assert c.post(f"/runs/{state.run_id}/resume", json={}).status_code == 202
        assert c.app.state.jobs.worker.wait_idle(30)
        assert kinds(c, state.run_id)[-1] == "done"
    assert env.llm.calls["ScriptOptions"] == 2  # one per run: the interrupted one's writer was not paid for again


# --- the event log -----------------------------------------------------------------------------------------------


def test_event_ids_grow_per_run_and_a_half_written_line_is_not_an_event(tmp_path):
    log = events.EventLog(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ids = [log.append("a", events.Queued(position=1, ahead=[])).id for _ in range(3)]
    assert ids == [1, 2, 3] and log.append("b", events.Queued(position=1, ahead=[])).id == 1
    with log.path("a").open("a", encoding="utf-8") as f:
        f.write('{"id": 4, "type": "queued", "posi')  # a crash mid-write
    assert [e.id for e in events.EventLog(tmp_path).after("a")] == [1, 2, 3]  # a fresh process reads the file
    assert [e.id for e in log.after("a", 1)] == [2, 3] and log.after("a", 3) == []


def test_the_events_are_typed_in_the_openapi_schema_the_frontend_types_come_from():
    schema = json.loads(api.openapi_json())
    union = schema["paths"]["/runs/{run_id}/log"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    names = {ref["$ref"].rsplit("/", 1)[1] for ref in union["items"]["oneOf"]}
    assert {"NodeStarted", "NodeFinished", "MediaMade", "ApprovalNeeded", "CriticReported", "BudgetStopped", "Failed",
            "Done"} <= names
    assert "id" in schema["components"]["schemas"]["NodeStarted"]["required"]  # `id: number`, not optional, in TypeScript
    types = {schema["components"]["schemas"][name]["properties"]["type"]["const"] for name in names}
    assert not types & {"open", "error", "message"}  # EventSource's own event names: a message called "error" fires onerror
    committed = REPO / "web" / "openapi.json"
    assert committed.read_text(encoding="utf-8") == api.openapi_json(), \
        "the API changed: run `npm run types` in web/ (it rewrites web/openapi.json and web/lib/api-schema.d.ts)"


def test_serve_listens_on_localhost_only(env, monkeypatch):
    import uvicorn
    from typer.testing import CliRunner

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    result = CliRunner().invoke(cli.app, ["serve", "--port", "8011", "--no-trace"])
    assert result.exit_code == 0, result.output
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8011
    assert seen["log_config"]["handlers"]["access"]["stream"] == "ext://sys.stderr"  # logs on stderr
    assert result.stdout == "" and "http://127.0.0.1:8011" in result.stderr and "mock" in result.stderr


def test_a_long_tool_error_keeps_its_first_words_where_ffmpeg_puts_the_cause():
    import subprocess as sp

    error = sp.CalledProcessError(3752568763, ["ffmpeg", "-i", "x"])
    error.add_note("ffmpeg said: [libx264 @ 0000024d] malloc of size 8294400 failed " + "x" * 400
                   + " Terminating thread with return code -542398533 (Generic error in an external library)")
    message, _, _ = cli.explain(error, "r1")
    assert "malloc of size 8294400 failed" in message and message.endswith("(Generic error in an external library)")
    assert len(message) < 400


# --- providers per role and OpenAI's free daily tokens (the OpenAI switch, 2026-10-26) -------------------------


def test_health_lists_every_roles_provider_and_todays_allowance_without_a_secret(client, env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SECRET-000")
    monkeypatch.setenv("FAL_KEY", "fal-SECRET-789")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_DATA_SHARING", "1")
    ledger.record(ledger.LedgerEntry(step="writer", model="gpt-5-mini-2025-08-07", prompt_tokens=800,
                                     completion_tokens=1200, cost_usd=0.0026), ledger.DEFAULT_PATH)
    h = client.get("/health")
    body = h.json()
    assert "SECRET" not in h.text
    assert (body["provider"], body["model"], body["critic_model"], body["key_set"]) == ("openai", "gpt-5-mini", "gpt-5.1", True)
    roles = {r["role"]: r for r in body["setup"]["roles"]}
    assert roles["text"]["model"] == "gpt-5-mini" and roles["embeddings"]["model"] == "gemini-embedding-2"
    assert roles["images (--real)"]["model"] == "gpt-image-2" and body["media"] == "mock"
    assert body["setup"]["keys"] == {"OPENAI_API_KEY": True, "GEMINI_API_KEY": True, "FAL_KEY": True, "YOUTUBE_API_KEY": False}
    assert body["setup"]["allowance"]["configured"]["OPENAI_DATA_SHARING"] is True
    mini = next(g for g in body["allowance"]["groups"] if g["group"] == "mini")
    assert (mini["tokens"], mini["left"], mini["billed_usd"]) == (2000, 2_498_000, 0)
    assert "data sharing" in body["billing"]


def test_the_web_stays_mock_whatever_the_image_settings_say(env, monkeypatch):
    monkeypatch.setenv("MOCK_MEDIA", "0")
    monkeypatch.setenv("IMAGE_PROVIDER", "openai")
    with TestClient(api.create_app(trace=False)) as c:
        run_id = start(c, approve="yes")
        settle(c)
        started = next(e for e in log_of(c, run_id) if e["type"] == "run_started")
        assert started["media"] == "mock" and api.os.environ["MOCK_MEDIA"] == "1"
        from shortsmith.media import images

        assert images.source().provider == "mock" and images.price_per_image() == 0
        assert "billed $0" in started["billing"]
