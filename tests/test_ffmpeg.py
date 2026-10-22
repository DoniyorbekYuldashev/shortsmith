"""ffmpeg.py: commands checked as pure lists of arguments; ffmpeg itself runs only in the live tests.

The Ken Burns tests don't compare filter strings: they evaluate the zoompan expressions for
every frame (Python understands + - * / and parentheses the same way ffmpeg does) and check
where the camera goes. Any correct expression passes, which is what the exercise needs too.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from shortsmith.media import ffmpeg
from shortsmith.media.tts import Voiceover, Word
from shortsmith.models import Script

UPSCALED_W, UPSCALED_H = 2160, 3840  # what zoompan sees as iw, ih after the 2x upscale


def zoompan_options(cmd: list[str]) -> dict[str, str]:
    """The zoompan filter's options, quotes removed: {"z": "...", "x": "...", "d": "90", ...}."""
    chain = next(arg for arg in cmd if "zoompan=" in arg)
    text = chain[chain.index("zoompan=") + len("zoompan="):]
    options, current, quoted = [], "", False
    for char in text:
        if char == "'":
            quoted = not quoted
        elif char in ":," and not quoted:
            options.append(current)
            current = ""
            if char == ",":  # the next filter in the chain starts here
                break
        else:
            current += char
    if current:
        options.append(current)
    return dict(option.split("=", 1) for option in options)


def evaluate(expression: str, on: int, zoom: float = 1.0) -> float:
    names = {"on": on, "zoom": zoom, "iw": UPSCALED_W, "ih": UPSCALED_H, "min": min, "max": max}
    return eval(expression, {"__builtins__": {}}, names)  # test-only: our own expressions


def camera(cmd: list[str]) -> list[tuple[float, float, float]]:
    """(zoom, x, y) for every output frame, the way zoompan computes them: z first, then x and y."""
    options = zoompan_options(cmd)
    path = []
    for on in range(int(options["d"])):
        zoom = evaluate(options["z"], on)
        path.append((zoom, evaluate(options["x"], on, zoom), evaluate(options["y"], on, zoom)))
    return path


def clip(duration_s: float = 3.0, motion: str = "zoom_in") -> list[str]:
    return ffmpeg.ken_burns_cmd(Path("scenes/scene_01.png"), Path("clips/scene_01.mp4"), duration_s, motion)


# --- Ken Burns: one image -> one clip -----------------------------------------------------


def test_frames_rounds_to_whole_frames():
    assert ffmpeg.frames(3.0) == 90
    assert ffmpeg.frames(2.51) == 75  # 75.3 frames -> 75
    assert ffmpeg.frames(0.001) == 1  # never an empty clip


def test_clip_reads_the_image_and_writes_the_output():
    cmd = clip()
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-i") + 1] == "scenes/scene_01.png"
    assert cmd[-1] == "clips/scene_01.mp4"


def test_duration_becomes_a_frame_count_at_30_fps():
    options = zoompan_options(clip(duration_s=3.0))
    assert (options["d"], options["fps"]) == ("90", "30")
    assert zoompan_options(clip(duration_s=2.51))["d"] == "75"


def test_output_is_vertical_full_hd_h264_yuv420p():
    cmd = clip()
    assert zoompan_options(cmd)["s"] == "1080x1920"
    assert "libx264" in cmd
    assert "yuv420p" in " ".join(cmd)  # phones can't play the 4:4:4 a PNG would give


def test_image_is_upscaled_to_2x_before_zoompan():
    chain = next(arg for arg in clip() if "zoompan=" in arg)
    assert "scale=2160:3840" in chain
    assert chain.index("scale=2160:3840") < chain.index("zoompan=")


def test_zoom_in_moves_in_around_the_centre():
    path = camera(clip(motion="zoom_in"))
    zooms = [zoom for zoom, _, _ in path]
    assert zooms[0] == pytest.approx(1.0) and zooms[-1] == pytest.approx(1.2)
    assert zooms == sorted(zooms)
    for zoom, x, y in path:
        assert x == pytest.approx((UPSCALED_W - UPSCALED_W / zoom) / 2)
        assert y == pytest.approx((UPSCALED_H - UPSCALED_H / zoom) / 2)


def test_zoom_out_is_zoom_in_backwards():
    path = camera(clip(motion="zoom_out"))
    zooms = [zoom for zoom, _, _ in path]
    assert zooms[0] == pytest.approx(1.2) and zooms[-1] == pytest.approx(1.0)
    assert zooms == sorted(zooms, reverse=True)
    zoom, x, _ = path[len(path) // 2]
    assert x == pytest.approx((UPSCALED_W - UPSCALED_W / zoom) / 2)


@pytest.mark.parametrize("motion, first, last", [("pan_right", "left", "right"), ("pan_left", "right", "left")])
def test_pans_cross_the_whole_picture_at_a_fixed_zoom(motion, first, last):
    path = camera(clip(motion=motion))
    zoom = path[0][0]
    assert zoom > 1.0  # zoomed in, or there is nothing to pan across
    assert all(z == pytest.approx(zoom) for z, _, _ in path)
    edge = {"left": 0.0, "right": UPSCALED_W - UPSCALED_W / zoom}
    assert path[0][1] == pytest.approx(edge[first])
    assert path[-1][1] == pytest.approx(edge[last])
    assert all(y == pytest.approx((UPSCALED_H - UPSCALED_H / zoom) / 2) for _, _, y in path)


@pytest.mark.parametrize("motion", ["zoom_in", "zoom_out", "pan_left", "pan_right"])
def test_window_never_leaves_the_picture(motion):
    for zoom, x, y in camera(clip(motion=motion)):
        assert 1.0 <= zoom <= 1.2 + 1e-9
        assert -1e-6 <= x <= UPSCALED_W - UPSCALED_W / zoom + 1e-6
        assert -1e-6 <= y <= UPSCALED_H - UPSCALED_H / zoom + 1e-6


def test_ai_clip_falls_back_to_zoom_in():
    assert zoompan_options(clip(motion="ai_clip")) == zoompan_options(clip(motion="zoom_in"))


def test_one_frame_clip_does_not_divide_by_zero():
    assert len(camera(clip(duration_s=1 / 30))) == 1


# --- the timeline: scenes, voice and subtitles share one frame count ----------------------


def _voiceover(name: str, last_word_end: float, file_s: float) -> Voiceover:
    words = [Word(text="hi", start_s=0.2, end_s=last_word_end)]
    return Voiceover(audio_path=Path(name), duration_s=file_s, words=words)


def test_scene_ends_at_its_last_word_plus_a_pause_not_at_the_file_end():
    voiceovers = [_voiceover("a.mp3", 2.0, 2.9), _voiceover("b.mp3", 3.0, 3.9)]  # ~0.9 s silent tails
    assert ffmpeg.scene_frames(voiceovers) == [69, 120]  # (2.0+0.3)*30, (3.0+0.3+0.7)*30: last scene holds longer


def test_scene_without_word_timings_uses_the_file_length():
    voiceover = Voiceover(audio_path=Path("a.mp3"), duration_s=1.0, words=[])
    assert ffmpeg.scene_frames([voiceover]) == [60]  # 1.0 + 0.3 pause + 0.7 end hold


def test_scene_starts_add_up_the_scenes_before():
    assert ffmpeg.scene_starts([69, 120, 30]) == pytest.approx([0.0, 2.3, 6.3])


# --- music, the final command, the runner ------------------------------------------------


def test_music_bed_is_generated_by_ffmpeg_not_downloaded():
    cmd = ffmpeg.music_bed_cmd(Path("music.wav"), 31.5)
    assert cmd[cmd.index("-f") + 1] == "lavfi"
    source = cmd[cmd.index("-i") + 1]
    assert source.startswith("aevalsrc=") and source.endswith(":d=31.500")
    assert not any("http" in arg for arg in cmd)
    assert cmd[-1] == "music.wav"


def _final(srt: Path = Path("subtitles.srt"), music: Path = Path("music.wav")) -> list[str]:
    voices = [Path("audio/scene_01.mp3"), Path("audio/scene_02.mp3")]
    return ffmpeg.final_cmd(Path("clips.txt"), voices, [69, 120], music, srt, Path("final.mp4"))


def _graph(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


def test_final_inputs_are_clips_then_voices_then_looped_music():
    cmd = _final()
    inputs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
    assert inputs == ["clips.txt", "audio/scene_01.mp3", "audio/scene_02.mp3", "music.wav"]
    assert cmd[cmd.index("clips.txt") - 3 : cmd.index("clips.txt")] == ["-f", "concat", "-i"]
    assert cmd[cmd.index("music.wav") - 3 : cmd.index("music.wav")] == ["-stream_loop", "-1", "-i"]


def test_each_voice_is_padded_and_cut_to_its_scene_in_samples():
    graph = _graph(_final())
    assert "[1:a]" in graph and "apad=whole_len=110400,atrim=end_sample=110400" in graph  # 69 frames x 1600
    assert "[2:a]" in graph and "apad=whole_len=192000,atrim=end_sample=192000" in graph  # 120 frames x 1600
    assert "concat=n=2:v=0:a=1" in graph
    assert "[3:a]" in graph and "atrim=end_sample=302400" in graph  # music cut to the whole video
    assert "afade=t=out:st=3.800:d=2.5" in graph  # 189 frames = 6.3 s, fade over the last 2.5 s


def test_music_is_quiet_and_ducked_under_the_voice():
    graph = _graph(_final())
    assert f"volume={ffmpeg.MUSIC_DB}dB" in graph
    assert "[bed][key]sidechaincompress" in graph  # the voice copy steers the music's volume
    assert "amix=inputs=2:duration=first:normalize=0" in graph


def test_subtitles_are_burned_in_big_white_and_outlined_in_the_lower_third():
    graph = _graph(_final())
    assert graph.startswith("[0:v]subtitles=subtitles.srt:force_style='")
    style = ffmpeg.subtitle_style()
    for setting in ("PlayResY=1920", "PrimaryColour=&H00FFFFFF", "OutlineColour=&H00000000", "Alignment=2"):
        assert setting in style
    assert ffmpeg.SUBTITLE_STYLE["MarginV"] > 1920 / 4  # above the platform buttons at the bottom


def test_subtitles_path_must_be_a_plain_relative_name():
    with pytest.raises(ValueError, match="cwd=run_dir"):
        _final(srt=Path("D:/All codes/shortsmith/runs/x/subtitles.srt"))  # drive colon + space break the filter


def test_final_output_plays_on_phones():
    cmd = _final()
    assert cmd[-1] == "final.mp4"
    for flag, value in [("-c:v", "libx264"), ("-pix_fmt", "yuv420p"), ("-c:a", "aac"), ("-movflags", "+faststart")]:
        assert cmd[cmd.index(flag) + 1] == value
    assert cmd[cmd.index("-frames:v") + 1] == "189"


def test_run_attaches_ffmpegs_own_message_when_it_fails(monkeypatch):
    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="scenes/scene_09.png: No such file or directory")

    monkeypatch.setattr(ffmpeg.subprocess, "run", failing_run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        ffmpeg.run(["ffmpeg", "-i", "scenes/scene_09.png"])
    assert "No such file" in error.value.__notes__[0]


# --- assemble: the whole edit, with ffmpeg faked --------------------------------------------


def _script(n: int = 4) -> Script:
    scenes = [
        {"idx": i, "narration": " ".join(["word"] * 18), "visual_prompt": "...", "duration_s": 5, "motion": "pan_left"}
        for i in range(1, n + 1)
    ]
    return Script.model_validate({"title": "t", "hook": "h", "scenes": scenes, "cta": "c", "language": "en"})


def _fake_run(monkeypatch) -> list[tuple[list[str], Path]]:
    calls = []
    monkeypatch.setattr(ffmpeg, "run", lambda cmd, cwd=None: calls.append((cmd, cwd)))
    return calls


def test_assemble_runs_every_command_inside_the_run_folder_with_relative_names(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "with space"  # like D:\All codes\...
    run_dir.mkdir(parents=True)
    voiceovers = [_voiceover(str(run_dir / "audio" / f"scene_{i:02d}.mp3"), 2.0, 2.9) for i in range(1, 5)]
    calls = _fake_run(monkeypatch)

    final = ffmpeg.assemble(run_dir, _script(), voiceovers)

    assert final == run_dir / "final.mp4"
    assert len(calls) == 4 + 1 + 1  # one clip per scene, the music bed, the final mix
    assert all(cwd == run_dir for _, cwd in calls)
    assert not any(str(tmp_path) in arg or "\\" in arg for cmd, _ in calls for arg in cmd)
    assert calls[0][0][-1] == "clips/scene_01.mp4"
    assert calls[4][0][-1] == "music.wav"
    final_cmd = calls[5][0]
    assert "audio/scene_01.mp3" in final_cmd and final_cmd[-1] == "final.mp4"
    assert (run_dir / "clips.txt").read_text(encoding="utf-8").splitlines()[1] == "file 'clips/scene_02.mp4'"


def test_assemble_writes_subtitles_on_the_video_timeline(monkeypatch, tmp_path):
    voiceovers = [_voiceover(str(tmp_path / f"scene_{i:02d}.mp3"), 2.0, 2.9) for i in range(1, 5)]
    _fake_run(monkeypatch)

    ffmpeg.assemble(tmp_path, _script(), voiceovers)

    srt = (tmp_path / "subtitles.srt").read_text(encoding="utf-8")
    assert "2\n00:00:02,500 --> 00:00:04,300\nhi\n" in srt  # scene 2 starts at 69 frames = 2.3 s; word 0.2-2.0 s


def test_assemble_uses_your_music_instead_of_generating_a_bed(monkeypatch, tmp_path):
    song = tmp_path / "my music" / "song.mp3"  # absolute, with a space: fine as an input
    voiceovers = [_voiceover(str(tmp_path / f"scene_{i:02d}.mp3"), 2.0, 2.9) for i in range(1, 5)]
    calls = _fake_run(monkeypatch)

    ffmpeg.assemble(tmp_path, _script(), voiceovers, music=song)

    assert len(calls) == 4 + 1  # no music.wav generated
    final_cmd = calls[-1][0]
    at = final_cmd.index(song.resolve().as_posix())
    assert final_cmd[at - 3 : at] == ["-stream_loop", "-1", "-i"]


# --- live: ffmpeg really renders (skipped where it isn't installed) --------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


@needs_ffmpeg
@pytest.mark.parametrize("size", [(576, 1024), (1080, 1920)], ids=["flux", "placeholder"])
def test_one_second_clip_renders_from_either_image_size(tmp_path, size):
    image, out = tmp_path / "scene.png", tmp_path / "clip.mp4"
    Image.new("RGB", size, (200, 120, 40)).save(image)

    ffmpeg.run(ffmpeg.ken_burns_cmd(image, out, 1.0, "pan_right"))

    info = ffmpeg.probe(out)
    assert (info.width, info.height, info.fps, info.frames) == (1080, 1920, 30, 30)
    assert (info.video_codec, info.pix_fmt, info.audio_codec) == ("h264", "yuv420p", None)
