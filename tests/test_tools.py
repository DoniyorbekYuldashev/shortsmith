"""agents/tools.py: schemas from the args models, argument validation, and each tool in a
temporary run folder (placeholder images are real; voice and ffmpeg are faked)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from shortsmith.agents import tools
from shortsmith.media import ffmpeg, tts
from shortsmith.media.tts import Voiceover, Word
from shortsmith.models import Script
from shortsmith.research import rag, youtube


def _script() -> Script:
    scenes = [
        {"idx": i, "narration": " ".join(["word"] * 15), "visual_prompt": f"prompt {i}", "duration_s": 6,
         "motion": "zoom_in"}
        for i in range(1, 5)
    ]
    return Script.model_validate({"title": "t", "hook": "h", "scenes": scenes, "cta": "c", "language": "en"})


@pytest.fixture
def ctx(tmp_path, monkeypatch) -> tools.ToolContext:
    monkeypatch.setenv("MOCK_MEDIA", "1")  # placeholders: free, and no fal key needed
    (tmp_path / "script.json").write_text(_script().model_dump_json(), encoding="utf-8")
    return tools.ToolContext.for_run(tmp_path)


def fake_synthesize(text: str, voice: str, out_path: Path) -> Voiceover:
    """edge-tts stand-in: an empty audio file and 0.3 s per word."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"mp3")
    words = [Word(text=w, start_s=i * 0.3, end_s=i * 0.3 + 0.25) for i, w in enumerate(text.split())]
    return Voiceover(audio_path=out_path, duration_s=len(words) * 0.3 + 0.4, words=words)


def call(ctx: tools.ToolContext, name: str, **arguments) -> dict:
    return json.loads(tools.call_tool(ctx, name, json.dumps(arguments)))


# --- schemas ---------------------------------------------------------------------------


def test_every_tool_is_an_openai_function_with_its_docstring_as_description():
    by_name = {s["function"]["name"]: s for s in tools.schemas()}
    assert set(by_name) == {"generate_image", "text_to_speech", "render_video"}
    for name, sent in by_name.items():
        assert sent["type"] == "function"
        assert sent["function"]["description"] == tools.TOOLS[name].description
        assert sent["function"]["description"].startswith(tools.TOOLS[name].fn.__doc__.split()[0])


def test_the_schema_comes_from_the_args_model_without_titles():
    params = tools.schema(tools.TOOLS["generate_image"])["function"]["parameters"]
    assert params == {
        "type": "object",
        "properties": {"scene": {"type": "integer", "minimum": 1,
                                 "description": "Scene number (idx) from the script, starting at 1."}},
        "required": ["scene"],
        "additionalProperties": False,  # extra="forbid": the API is told, and validation enforces it
    }
    assert '"title"' not in json.dumps(tools.schemas())  # Pydantic's auto titles cost tokens every turn


def test_a_property_called_title_survives_the_title_cleanup():
    class Named(BaseModel):
        title: str

    params = tools._without_titles(Named.model_json_schema())
    assert params["properties"] == {"title": {"type": "string"}}
    assert params["required"] == ["title"]


def test_a_tool_without_arguments_has_an_empty_object_schema():
    params = tools.schema(tools.TOOLS["render_video"])["function"]["parameters"]
    assert params == {"type": "object", "properties": {}, "additionalProperties": False}


# --- validation: the model's arguments are untrusted input ---------------------------------


def test_an_unknown_tool_names_the_real_ones(ctx):
    with pytest.raises(ValueError, match="unknown tool 'make_coffee'; the tools are: generate_image, text_to_speech"):
        tools.call_tool(ctx, "make_coffee", "{}")


@pytest.mark.parametrize(
    "arguments", ['{"scene": "two"}', '{"scene": 0}', '{}', '{"scene": 1, "style": "noir"}', "scene=1"]
)
def test_bad_arguments_raise_a_validation_error(ctx, arguments):
    with pytest.raises(ValidationError):
        tools.call_tool(ctx, "generate_image", arguments)


def test_a_scene_that_does_not_exist_is_an_error(ctx):
    with pytest.raises(ValueError, match="no scene 9: this script has scenes 1-4"):
        call(ctx, "generate_image", scene=9)


def test_empty_arguments_mean_no_arguments(ctx, monkeypatch):
    monkeypatch.setattr(tools.ffmpeg, "assemble", lambda *a: pytest.fail("rendered without images"))
    with pytest.raises(FileNotFoundError):
        tools.call_tool(ctx, "render_video", "")  # some models send "" instead of "{}" for no arguments


# --- the tools --------------------------------------------------------------------------


def test_generate_image_makes_the_scene_image_and_answers_in_short_json(ctx):
    raw = tools.call_tool(ctx, "generate_image", '{"scene": 2}')

    assert json.loads(raw) == {"ok": True, "scene": 2, "path": "scenes/scene_02.png", "reused": False}
    assert len(raw) < 100  # every result is re-sent to the model on every later turn
    assert (ctx.run_dir / "scenes" / "scene_02.png").exists()


def test_an_existing_image_is_never_made_again(ctx, monkeypatch):
    (ctx.run_dir / "scenes").mkdir()
    (ctx.run_dir / "scenes" / "scene_01.png").write_bytes(b"png")
    monkeypatch.setattr(tools.images, "generate_image", lambda *a, **k: pytest.fail("paid twice"))

    assert call(ctx, "generate_image", scene=1)["reused"] is True


def test_voice_is_saved_for_make_run_once_every_scene_has_it(ctx, monkeypatch):
    monkeypatch.setattr(tools.tts, "synthesize", fake_synthesize)

    for idx in (1, 2, 3):
        result = call(ctx, "text_to_speech", scene=idx)
    assert result == {"ok": True, "scene": 3, "path": "audio/scene_03.mp3", "duration_s": 4.9, "words": 15}
    assert not (ctx.run_dir / tts.VOICEOVERS_FILE).exists()  # incomplete: `make --run` would voice again anyway

    call(ctx, "text_to_speech", scene=4)
    saved = tts.load_voiceovers(ctx.run_dir)
    assert [v.audio_path.name for v in saved] == ["scene_01.mp3", "scene_02.mp3", "scene_03.mp3", "scene_04.mp3"]


def test_render_video_says_exactly_what_is_missing(ctx, monkeypatch):
    monkeypatch.setattr(tools.tts, "synthesize", fake_synthesize)
    call(ctx, "generate_image", scene=1)
    call(ctx, "text_to_speech", scene=2)

    with pytest.raises(FileNotFoundError) as error:
        call(ctx, "render_video")
    assert str(error.value) == (
        "not ready to render: no image for scenes [2, 3, 4]; no voiceover for scenes [1, 3, 4]"
    )


def test_render_video_assembles_when_everything_is_there(ctx, monkeypatch):
    monkeypatch.setattr(tools.tts, "synthesize", fake_synthesize)
    for idx in range(1, 5):
        call(ctx, "generate_image", scene=idx)
        call(ctx, "text_to_speech", scene=idx)
    seen = {}

    def fake_assemble(run_dir, script, voiceovers):
        seen["voices"] = [v.audio_path.name for v in voiceovers]
        (run_dir / "final.mp4").write_bytes(b"mp4")
        return run_dir / "final.mp4"

    monkeypatch.setattr(tools.ffmpeg, "assemble", fake_assemble)
    monkeypatch.setattr(tools.ffmpeg, "probe", lambda path: ffmpeg.VideoInfo(
        width=1080, height=1920, fps=30, frames=903, duration_s=30.1, size_bytes=2_800_000,
        video_codec="h264", pix_fmt="yuv420p", audio_codec="aac"))

    assert call(ctx, "render_video") == {"ok": True, "path": "final.mp4", "duration_s": 30.1, "size_mb": 2.8, "scenes": 4}
    assert seen["voices"] == ["scene_01.mp3", "scene_02.mp3", "scene_03.mp3", "scene_04.mp3"]  # in scene order


def test_fail_first_fails_only_the_first_call_of_that_tool(ctx):
    ctx.fail_first = {"generate_image"}

    with pytest.raises(ConnectionError):
        call(ctx, "generate_image", scene=1)
    assert call(ctx, "generate_image", scene=1)["ok"] is True
    assert call(ctx, "generate_image", scene=2)["ok"] is True


def test_fail_first_still_validates_arguments_first(ctx):
    ctx.fail_first = {"generate_image"}
    with pytest.raises(ValidationError):
        call(ctx, "generate_image", scene="x")
    assert ctx.failed == set()  # the injected failure is still waiting for a valid call


def test_a_run_with_a_complete_saved_voice_does_not_need_voicing_again(ctx, monkeypatch):
    monkeypatch.setattr(tools.tts, "synthesize", fake_synthesize)
    for idx in range(1, 5):
        call(ctx, "text_to_speech", scene=idx)

    again = tools.ToolContext.for_run(ctx.run_dir)
    assert sorted(again.voiceovers) == [1, 2, 3, 4]


def test_a_half_voiced_run_keeps_each_scenes_voice_on_disk(ctx, monkeypatch):
    """The MCP server loads a fresh context for every call: voice made earlier must come from disk."""
    monkeypatch.setattr(tools.tts, "synthesize", fake_synthesize)
    for idx in (1, 2, 3):
        call(ctx, "text_to_speech", scene=idx)
    (ctx.run_dir / tools.audio_file(3)).unlink()  # timings without their audio don't count

    again = tools.ToolContext.for_run(ctx.run_dir)
    assert sorted(again.voiceovers) == [1, 2]
    assert again.voiceovers[2].audio_path == ctx.run_dir / "audio" / "scene_02.mp3"
    assert again.voiceovers[2].words == ctx.voiceovers[2].words
    assert tools.run_status(again) == tools.RunStatus(images=[], voiceovers=[1, 2], final=False)


# --- research tools (Phase 5): no run folder ---------------------------------------------------


def _trend_report(query: str) -> youtube.TrendReport:
    when = datetime(2026, 10, 23, tzinfo=timezone.utc)
    clip = youtube.TrendVideo(video_id="abc", title="Cozy cafe tour", channel="c", views=12_000, likes=300,
                              comments=4, published=when, duration_s=41, vertical=True)
    return youtube.TrendReport(query=query, language="en", published_after=when - timedelta(days=30), fetched_at=when,
                               candidates=5, shorts_length=1, videos=[clip], from_cache=True, units_used=0)


def _hit(id: str, kind: str = "hook") -> rag.Retrieved:
    return rag.Retrieved(id=id, kind=kind, title=id, group="g", text="Template: ...", similarity=0.7)


def test_research_tools_are_registered_apart_from_the_media_tools():
    assert tools.names("media") == ["generate_image", "text_to_speech", "render_video"]
    assert tools.names("research") == ["search_trends", "retrieve_hooks"]
    research = {s["function"]["name"]: s["function"] for s in tools.schemas("research")}
    assert set(research) == {"search_trends", "retrieve_hooks"}
    assert research["retrieve_hooks"]["parameters"]["properties"]["k"] == {
        "type": "integer", "minimum": 1, "maximum": 10, "default": 5, "description": "How many hook patterns to return."}
    assert '"title"' not in json.dumps(research)


def test_search_trends_needs_no_run_and_returns_the_report(monkeypatch):
    seen = []
    monkeypatch.setattr(tools.youtube, "search_trends", lambda q, lang: seen.append((q, lang)) or _trend_report(q))
    result = json.loads(tools.call_tool(None, "search_trends", '{"query": "coffee Tashkent"}'))
    assert seen == [("coffee Tashkent", "en")]
    assert (result["ok"], result["since"], result["from_cache"], result["units_used"]) == (True, "2026-09-23", True, 0)
    assert result["videos"][0]["url"] == "https://www.youtube.com/watch?v=abc"


def test_retrieve_hooks_adds_brand_rules_only_for_a_brand(monkeypatch):
    calls = []

    def fake_retrieve(query, k, kind="hook", brand=None):
        calls.append((query, k, kind, brand))
        return [_hit(f"{kind}-{i}", kind) for i in range(k)]

    monkeypatch.setattr(tools.rag, "retrieve", fake_retrieve)
    plain = tools.run_tool(None, "retrieve_hooks", '{"query": "plov", "k": 2}')
    assert [h.id for h in plain.hooks] == ["hook-0", "hook-1"] and plain.brand_rules == []
    branded = tools.run_tool(None, "retrieve_hooks", '{"query": "plov", "brand": "anor-va-dolchin"}')
    assert len(branded.hooks) == 5 and [r.kind for r in branded.brand_rules] == ["brand"] * 3
    assert calls[-2] == ("plov: tone, words to use and avoid, visual style", 3, "brand", "anor-va-dolchin")


def test_research_arguments_are_validated_like_any_other(monkeypatch):
    monkeypatch.setattr(tools.rag, "retrieve", lambda *a, **k: pytest.fail("ran with bad arguments"))
    with pytest.raises(ValidationError):
        tools.run_tool(None, "retrieve_hooks", '{"query": "plov", "k": 50}')
    with pytest.raises(ValidationError):
        tools.run_tool(None, "search_trends", '{"query": "x", "language": "de"}')


def test_a_media_tool_without_a_run_is_refused():
    with pytest.raises(ValueError, match="generate_image works inside a run"):
        tools.run_tool(None, "generate_image", '{"scene": 1}')
