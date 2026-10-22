"""make.py's own logic (what gets reused, what gets warned about), with every media step faked."""

from pathlib import Path

import pytest

from shortsmith import llm, make
from shortsmith.media import ffmpeg
from shortsmith.media.tts import Voiceover, Word
from shortsmith.models import NARRATION_WORDS, Script


def _script(language: str = "en") -> Script:
    scenes = [
        {"idx": i, "narration": " ".join(["word"] * 15), "visual_prompt": f"prompt {i}", "duration_s": 6,
         "motion": "zoom_in"}
        for i in range(1, 5)
    ]
    return Script.model_validate({"title": "t", "hook": "h", "scenes": scenes, "cta": "c", "language": language})


def _voiceovers(run_dir: Path, last_word_end: float = 7.0) -> list[Voiceover]:
    return [
        Voiceover(audio_path=run_dir / f"{i}.mp3", duration_s=last_word_end + 0.5,
                  words=[Word(text="word", start_s=0.2, end_s=last_word_end)])
        for i in range(1, 5)
    ]


def test_existing_images_are_never_made_again(monkeypatch, tmp_path):
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "scene_02.png").write_bytes(b"png")
    made = []
    monkeypatch.setattr(make.images, "generate_image", lambda prompt, path, label: made.append(path.name))

    make.make_images(_script(), tmp_path)

    assert made == ["scene_01.png", "scene_03.png", "scene_04.png"]  # FLUX would charge for each one


def test_saved_voiceovers_are_reused_when_they_match_the_script(monkeypatch, tmp_path):
    saved = _voiceovers(tmp_path)
    monkeypatch.setattr(make.tts, "load_voiceovers", lambda run_dir: saved)
    monkeypatch.setattr(make.tts, "voice_script", lambda script, run_dir: pytest.fail("voiced again"))
    assert make.make_voice(_script(), tmp_path) is saved


def test_voice_is_made_again_when_the_saved_one_does_not_match(monkeypatch, tmp_path):
    fresh = _voiceovers(tmp_path)
    monkeypatch.setattr(make.tts, "load_voiceovers", lambda run_dir: fresh[:2])  # from an older 2-scene script
    monkeypatch.setattr(make.tts, "voice_script", lambda script, run_dir: fresh)
    assert make.make_voice(_script(), tmp_path) is fresh


def _probe(duration_s: float):
    return lambda path: ffmpeg.VideoInfo(
        width=1080, height=1920, fps=30, frames=round(duration_s * 30), duration_s=duration_s,
        size_bytes=3_000_000, video_codec="h264", pix_fmt="yuv420p", audio_codec="aac",
    )


def test_summary_shows_planned_next_to_real_and_on_screen(monkeypatch, tmp_path):
    monkeypatch.setattr(make.ffmpeg, "probe", _probe(30.3))
    text = make.summary(_script(), _voiceovers(tmp_path), tmp_path / "final.mp4")
    assert "zoom_in" in text and "6.0s" in text and "7.5s" in text  # planned 6.0, voice file 7.5
    assert "total" in text and "60" in text  # 4 x 15 words
    assert "WARNING" not in text


def test_summary_warns_when_the_video_misses_30_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(make.ffmpeg, "probe", _probe(15.2))
    text = make.summary(_script(), _voiceovers(tmp_path, last_word_end=3.0), tmp_path / "final.mp4")
    assert "WARNING: 15.2 s is outside the 25-35 s target" in text


def test_writer_prompt_states_the_same_word_targets_the_validator_checks():
    for low, high in NARRATION_WORDS.values():
        assert f"{low}-{high} words in total" in llm.WRITER_SYSTEM_PROMPT
    assert "Uzbek (Latin script): 9-10 words per scene" in llm.WRITER_SYSTEM_PROMPT  # 55-60 over 6 scenes


def test_a_topic_or_a_run_folder_is_required(monkeypatch):
    monkeypatch.setattr("sys.argv", ["make"])
    with pytest.raises(SystemExit):
        make.main()
