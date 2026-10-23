"""The Producer's make-like manifest, the hero clip (tool, mock, real path faked) and ffmpeg's clip support."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from shortsmith import ledger
from shortsmith.agents import producer, tools
from shortsmith.agents.director import HeroPlan
from shortsmith.media import animate, ffmpeg, images, tts
from shortsmith.media.tts import Voiceover, Word
from shortsmith.models import Script


def plan(prompt3: str = "picture 3", narration2: str = "words " * 13, hero_motion: str = "ai_clip") -> Script:
    scenes = [{"idx": i, "narration": narration2 if i == 2 else "words " * 13,
               "visual_prompt": prompt3 if i == 3 else f"picture {i}", "duration_s": 5,
               "motion": hero_motion if i == 1 else "zoom_in"} for i in range(1, 7)]
    return Script.model_validate({"title": "t", "hook": "h", "scenes": scenes, "cta": "c", "language": "en"},
                                 context={"saved": True})


HERO = HeroPlan(scene=1, motion_prompt="steam rises", reason="hook", fallback="zoom_out", cost_usd=0.3, approved=True)


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """A fake call_tool that writes the files the real tools would."""
    seen: list[tuple[str, int]] = []

    def call_tool(name: str, arguments: str) -> str:
        idx = json.loads(arguments)["scene"]
        seen.append((name, idx))
        path = {"generate_image": tools.image_file(idx), "text_to_speech": tools.audio_file(idx),
                "animate_image": ffmpeg.ai_clip_file(idx)}[name]
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(name.encode())
        return "{}"

    return NS(call=call_tool, seen=seen, dir=tmp_path)


def test_the_first_pass_makes_everything_and_the_second_nothing(calls):
    made = producer.produce(calls.dir, plan(), HERO, calls.call)
    assert str(made) == "images 1, 2, 3, 4, 5, 6; voices 1, 2, 3, 4, 5, 6; clip 1"
    assert calls.seen[-1] == ("animate_image", 1) and len(calls.seen) == 13
    assert Script.load(calls.dir / "script.json") == plan()  # the tools read this round's scenes from disk

    again = producer.produce(calls.dir, plan(), HERO, calls.call)
    assert str(again) == "nothing (all up to date)" and len(calls.seen) == 13


def test_a_new_picture_prompt_remakes_only_that_image_and_keeps_the_old_one(calls):
    producer.produce(calls.dir, plan(), HERO, calls.call)
    made = producer.produce(calls.dir, plan(prompt3="a better picture 3"), HERO, calls.call, round_=1)
    assert (made.images, made.voices, made.clips) == ([3], [], [])
    assert (calls.dir / "history" / "round_1" / "scenes" / "scene_03.png").read_bytes() == b"generate_image"


def test_new_words_remake_only_that_voice(calls):
    producer.produce(calls.dir, plan(), HERO, calls.call)
    made = producer.produce(calls.dir, plan(narration2="other words " * 7), HERO, calls.call, round_=1)
    assert (made.images, made.voices) == ([], [2])


def test_a_deleted_file_is_made_again_even_if_the_manifest_says_done(calls):
    producer.produce(calls.dir, plan(), HERO, calls.call)
    (calls.dir / tools.image_file(5)).unlink()
    assert producer.produce(calls.dir, plan(), HERO, calls.call).images == [5]


def test_the_clip_is_needed_only_when_approved_and_out_of_date(calls):
    assert producer.clip_needed(calls.dir, plan(), HERO)
    assert not producer.clip_needed(calls.dir, plan(), HERO.model_copy(update={"approved": False}))
    assert not producer.clip_needed(calls.dir, plan(hero_motion="zoom_out"), HERO)  # refused: Ken Burns
    assert not producer.clip_needed(calls.dir, plan(), None)
    producer.produce(calls.dir, plan(), HERO, calls.call)
    assert not producer.clip_needed(calls.dir, plan(), HERO)  # made and still fresh
    assert producer.clip_needed(calls.dir, plan(), HERO.model_copy(update={"motion_prompt": "leaves fall"}))


def test_not_approved_yet_means_no_clip(calls):
    made = producer.produce(calls.dir, plan(), HERO.model_copy(update={"approved": None}), calls.call)
    assert made.clips == []


# --- the animate_image tool and media/animate.py ----------------------------------------------------------


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_MEDIA", "1")
    (tmp_path / "script.json").write_text(plan().model_dump_json(), encoding="utf-8")
    return tools.ToolContext.for_run(tmp_path)


def test_animate_image_needs_the_scene_image_first(ctx):
    with pytest.raises(FileNotFoundError, match="call generate_image first"):
        tools.call_tool(ctx, "animate_image", json.dumps({"scene": 1, "motion": "steam rises"}))


def test_animate_image_makes_a_clip_once(ctx, monkeypatch):
    made = []
    def fake_animate(image, motion, out, step):
        made.append(motion)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"c")

    monkeypatch.setattr(animate, "animate_image", fake_animate)
    (ctx.run_dir / "scenes").mkdir()
    (ctx.run_dir / "scenes" / "scene_01.png").write_bytes(b"png")
    first = json.loads(tools.call_tool(ctx, "animate_image", json.dumps({"scene": 1, "motion": "steam rises"})))
    second = json.loads(tools.call_tool(ctx, "animate_image", json.dumps({"scene": 1, "motion": "steam rises"})))
    assert first == {"ok": True, "scene": 1, "path": "clips_ai/scene_01.mp4", "seconds": 5, "reused": False, "mock": True}
    assert second["reused"] is True and made == ["steam rises"]


def test_the_mock_clip_command_moves_unlike_ken_burns():
    cmd = animate.mock_clip_cmd(Path("in.png"), Path("out.mp4"))
    vf = cmd[cmd.index("-vf") + 1]
    assert "hue=h=40*t" in vf and "rotate=" in vf and "zoompan" not in vf
    assert cmd[cmd.index("-t") + 1] == "5" and cmd[-1] == "out.mp4"


def test_the_mock_clip_is_a_real_5_second_video(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_MEDIA", "1")
    images.generate_image("a cup of tea", tmp_path / "s.png", "Scene 1")
    out = animate.animate_image(tmp_path / "s.png", "steam rises", tmp_path / "clip.mp4")
    info = ffmpeg.probe(out)
    assert (info.width, info.height, info.fps) == (540, 960, 30) and 4.9 <= info.duration_s <= 5.1


def test_the_real_clip_path_records_the_price_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_MEDIA", "0")
    order, recorded = [], []
    fake_fal = NS(upload_file=lambda path: order.append("upload") or "https://fal.example/in.png",
                  subscribe=lambda model, arguments: order.append(("subscribe", model, arguments)) or
                  {"video": {"url": "https://fal.example/out.mp4"}})
    monkeypatch.setattr(animate, "fal_client", fake_fal)
    monkeypatch.setattr(animate.ledger, "record", lambda entry, path: order.append("ledger") or recorded.append(entry))
    monkeypatch.setattr(animate.urllib.request, "urlretrieve", lambda url, out: order.append("download"))

    animate.animate_image(tmp_path / "s.png", "steam rises", tmp_path / "clip.mp4")

    assert order[0] == "upload" and order[2:] == ["ledger", "download"]
    _, model, arguments = order[1]
    assert model == animate.CLIP_MODEL and arguments["aspect_ratio"] == "9:16" and arguments["duration"] == "5"
    assert recorded[0].cost_usd == ledger.MEDIA_PRICES[animate.CLIP_MODEL] == 0.30


# --- ffmpeg: a hero clip in the edit ----------------------------------------------------------------------


def test_the_clip_command_fits_a_clip_to_its_scene():
    cmd = ffmpeg.clip_cmd(Path("clips_ai/scene_01.mp4"), Path("clips/scene_01.mp4"), 6.0)
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"  # a 5-s clip under a 6-s voice loops
    assert cmd[cmd.index("-frames:v") + 1] == "180" and "-an" in cmd
    assert "crop=1080:1920" in cmd[cmd.index("-vf") + 1]


def test_assemble_uses_the_clip_for_an_ai_clip_scene_and_ken_burns_otherwise(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(ffmpeg, "run", lambda cmd, cwd=None: ran.append(cmd))
    monkeypatch.setattr(ffmpeg, "subtitles", lambda voiceovers, starts: "")
    (tmp_path / "clips_ai").mkdir()
    (tmp_path / "clips_ai" / "scene_01.mp4").write_bytes(b"clip")
    voices = [Voiceover(audio_path=tmp_path / f"{i}.mp3", duration_s=4, words=[Word(text="w", start_s=0, end_s=3.5)])
              for i in range(6)]
    ffmpeg.assemble(tmp_path, plan(), voices)
    first, second = ran[0], ran[1]
    assert "clips_ai/scene_01.mp4" in first and "zoompan" not in " ".join(first)
    assert "zoompan" in " ".join(second)


def test_an_ai_clip_scene_without_its_clip_falls_back_to_ken_burns(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(ffmpeg, "run", lambda cmd, cwd=None: ran.append(cmd))
    voices = [Voiceover(audio_path=tmp_path / f"{i}.mp3", duration_s=4, words=[Word(text="w", start_s=0, end_s=3.5)])
              for i in range(6)]
    ffmpeg.assemble(tmp_path, plan(), voices)
    assert "zoompan" in " ".join(ran[0])
    assert tts.VOICES["en"]
