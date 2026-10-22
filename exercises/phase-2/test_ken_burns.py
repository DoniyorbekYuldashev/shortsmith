"""Checks for the Step 2.3 exercise: the same assertions as tests/test_ffmpeg.py's Ken Burns tests.

They don't compare your filter string with ours: they evaluate your zoompan expressions for
every frame and check where the camera goes, so any correct expression passes.
Run from the repo root:  uv run pytest exercises/phase-2
"""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from ken_burns import ken_burns_cmd  # pytest puts this folder on sys.path
from shortsmith.media.ffmpeg import probe, run

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
    return eval(expression, {"__builtins__": {}}, names)  # test-only: your own expressions


def camera(cmd: list[str]) -> list[tuple[float, float, float]]:
    """(zoom, x, y) for every output frame, the way zoompan computes them: z first, then x and y."""
    options = zoompan_options(cmd)
    path = []
    for on in range(int(options["d"])):
        zoom = evaluate(options["z"], on)
        path.append((zoom, evaluate(options["x"], on, zoom), evaluate(options["y"], on, zoom)))
    return path


def clip(duration_s: float = 3.0, motion: str = "zoom_in") -> list[str]:
    return ken_burns_cmd(Path("scenes/scene_01.png"), Path("clips/scene_01.mp4"), duration_s, motion)


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
    assert "yuv420p" in " ".join(cmd)


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("size", [(576, 1024), (1080, 1920)], ids=["flux", "placeholder"])
def test_one_second_clip_renders_from_either_image_size(tmp_path, size):
    image, out = tmp_path / "scene.png", tmp_path / "clip.mp4"
    Image.new("RGB", size, (200, 120, 40)).save(image)

    run(ken_burns_cmd(image, out, 1.0, "pan_right"))  # ffmpeg itself judges your command here

    info = probe(out)
    assert (info.width, info.height, info.fps, info.frames) == (1080, 1920, 30, 30)
    assert (info.video_codec, info.pix_fmt, info.audio_codec) == ("h264", "yuv420p", None)
