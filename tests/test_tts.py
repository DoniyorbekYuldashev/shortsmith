"""tts.py without the network: subtitle maths for real, edge-tts and synthesize() faked."""

import asyncio
import shutil
from pathlib import Path

import pytest

from shortsmith.media import tts
from shortsmith.media.tts import Voiceover, Word
from shortsmith.models import Script


def _words(*spans: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, start_s=s, end_s=e) for t, s, e in spans]


def test_srt_time_uses_a_comma_before_milliseconds():
    assert tts.srt_time(0) == "00:00:00,000"
    assert tts.srt_time(62.5) == "00:01:02,500"
    assert tts.srt_time(3661.25) == "01:01:01,250"


def test_to_srt_groups_words_into_numbered_blocks():
    words = _words(("Find", 0.1, 0.4), ("the", 0.4, 0.5), ("coziest", 0.5, 1.0), ("cafe", 1.0, 1.4), ("today", 1.5, 2.0))
    assert tts.to_srt(tts.group_words(words, per_line=4)) == (
        "1\n00:00:00,100 --> 00:00:01,400\nFind the coziest cafe\n"
        "\n"
        "2\n00:00:01,500 --> 00:00:02,000\ntoday\n"
    )


def test_shift_moves_words_to_the_scene_start():
    shifted = tts.shift(_words(("Salom", 0.1, 0.5)), by_s=4.0)
    assert (shifted[0].start_s, shifted[0].end_s) == (4.1, 4.5)


def test_speak_writes_audio_and_converts_ticks(monkeypatch, tmp_path):
    class FakeCommunicate:
        def __init__(self, text, voice, boundary):
            assert boundary == "WordBoundary"  # word timings must be asked for explicitly

        async def stream(self):
            yield {"type": "audio", "data": b"ID3"}
            yield {"type": "WordBoundary", "offset": 1_250_000, "duration": 5_000_000, "text": "Toshkentda"}
            yield {"type": "audio", "data": b"..."}

    monkeypatch.setattr(tts.edge_tts, "Communicate", FakeCommunicate)
    out = tmp_path / "scene.mp3"

    words = asyncio.run(tts._speak("Toshkentda", "uz-UZ-MadinaNeural", out))

    assert out.read_bytes() == b"ID3..."
    assert words == [Word(text="Toshkentda", start_s=0.125, end_s=0.625)]


def test_synthesize_retries_when_no_audio_arrives(monkeypatch, tmp_path):
    attempts = []

    async def flaky_speak(text, voice, out_path):
        attempts.append(1)
        if len(attempts) < 3:
            raise tts.edge_tts.exceptions.NoAudioReceived("no audio")
        return []

    monkeypatch.setattr(tts, "_speak", flaky_speak)
    monkeypatch.setattr(tts, "audio_duration", lambda path: 1.5)
    monkeypatch.setattr(tts.time, "sleep", lambda seconds: None)  # tests shouldn't really wait

    voiceover = tts.synthesize("hi", "en-US-AndrewNeural", tmp_path / "a.mp3")

    assert (len(attempts), voiceover.duration_s) == (3, 1.5)


def test_synthesize_gives_up_after_the_last_attempt(monkeypatch, tmp_path):
    async def always_silent(text, voice, out_path):
        raise tts.edge_tts.exceptions.NoAudioReceived("no audio")

    monkeypatch.setattr(tts, "_speak", always_silent)
    monkeypatch.setattr(tts.time, "sleep", lambda seconds: None)
    with pytest.raises(tts.edge_tts.exceptions.NoAudioReceived):
        tts.synthesize("hi", "en-US-AndrewNeural", tmp_path / "a.mp3")


def test_subtitles_follow_the_video_timeline(tmp_path):
    voiceovers = [
        Voiceover(audio_path=tmp_path / "1.mp3", duration_s=2.9, words=_words(("Salom", 0.2, 0.8), ("do'stlar", 0.8, 1.6))),
        Voiceover(audio_path=tmp_path / "2.mp3", duration_s=2.9, words=_words(("Kofe", 0.2, 0.7))),
    ]
    srt = tts.subtitles(voiceovers, scene_starts=[0.0, 1.9])  # 1.9 = scene 1's last word + a pause

    assert "2\n00:00:02,100 --> 00:00:02,600\nKofe\n" in srt  # its own line, moved to where scene 2 starts
    assert srt.count(" --> ") == 2  # grouped per scene: no line spans a cut


def _uzbek_script() -> Script:
    scenes = [
        {"idx": i, "narration": f"sahna {i} " + " ".join(["so'z"] * 9), "visual_prompt": "...", "duration_s": 5,
         "motion": "zoom_in"}
        for i in range(1, 6)
    ]  # 5 x 11 = 55 words, an Uzbek ~30 s
    return Script.model_validate({"title": "t", "hook": "Salom", "scenes": scenes, "cta": "c", "language": "uz"})


def test_voice_script_saves_word_timings_that_load_back(monkeypatch, tmp_path):
    voices_used = []

    def fake_synthesize(text: str, voice: str, out_path: Path) -> Voiceover:
        voices_used.append(voice)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ID3")
        return Voiceover(audio_path=out_path, duration_s=2.0, words=_words((text, 0.1, 1.0)))

    monkeypatch.setattr(tts, "synthesize", fake_synthesize)

    voiceovers = tts.voice_script(_uzbek_script(), tmp_path)

    assert set(voices_used) == {"uz-UZ-MadinaNeural"}
    assert tts.load_voiceovers(tmp_path) == voiceovers  # re-assembly needs no edge-tts call


def test_a_copied_run_folder_uses_its_own_audio(monkeypatch, tmp_path):
    def fake_synthesize(text: str, voice: str, out_path: Path) -> Voiceover:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ID3")
        return Voiceover(audio_path=out_path, duration_s=2.0, words=_words((text, 0.1, 1.0)))

    monkeypatch.setattr(tts, "synthesize", fake_synthesize)
    tts.voice_script(_uzbek_script(), tmp_path / "original")
    shutil.copytree(tmp_path / "original", tmp_path / "copy")

    voiceovers = tts.load_voiceovers(tmp_path / "copy")

    assert voiceovers[0].audio_path == tmp_path / "copy" / "audio" / "scene_01.mp3"  # not the original's file
    assert "original" not in (tmp_path / "copy" / tts.VOICEOVERS_FILE).read_text(encoding="utf-8")


def test_load_voiceovers_gives_none_when_anything_is_missing(tmp_path):
    assert tts.load_voiceovers(tmp_path) is None  # never voiced
    voiceover = Voiceover(audio_path=Path("audio/scene_01.mp3"), duration_s=1.0, words=[])  # as saved
    (tmp_path / tts.VOICEOVERS_FILE).write_bytes(tts._VOICEOVERS.dump_json([voiceover]))
    assert tts.load_voiceovers(tmp_path) is None  # timings saved, but the audio file is gone
