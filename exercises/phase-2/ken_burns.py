"""🧠 Exercise, Step 2.3: build the Ken Burns ffmpeg command for one still image.

The finished version is ken_burns_cmd() in src/shortsmith/media/ffmpeg.py: try this one
first, then compare. Check it from the repo root:  uv run pytest exercises/phase-2
"""

from pathlib import Path

from shortsmith.media.ffmpeg import FPS, HEIGHT, UPSCALE, WIDTH, ZOOM  # 30, 1920, 2, 1080, 1.2
from shortsmith.models import Motion


def ken_burns_cmd(image: Path, out: Path, duration_s: float, motion: Motion) -> list[str]:
    """ffmpeg arguments that turn one still image into a silent moving clip of `duration_s`.

    Return a list like ["ffmpeg", "-y", "-i", <image>, "-vf", <filters>, ..., <out>]: a list,
    never one string, so no shell ever splits it (the repo path has a space in it).

    Hints, in order:
    1. Frames: n = round(duration_s * FPS), at least 1. 3.0 s -> 90 frames.
    2. Upscale first, to UPSCALE x the output size (2160x3840):
       "scale=2160:3840:force_original_aspect_ratio=increase" then "crop=2160:3840".
       FLUX gives 576x1024, placeholders are 1080x1920: after this both are the same size.
       Why: zoompan moves in whole pixels; on a 2x picture one step is half an output pixel.
    3. Then one zoompan: "zoompan=z='...':x='...':y='...':d=<n>:s=1080x1920:fps=30".
       One input frame in, d frames out. Put each expression in '...' quotes; join the
       filters with "," into one -vf string.
    4. Inside the expressions: `on` = output frame number (0 .. n-1), `iw`/`ih` = the input
       size, `zoom` = this frame's zoom (ffmpeg computes z first, then x and y).
       progress = on/(n-1) runs from 0 to 1 (careful when n is 1). The visible window is
       iw/zoom wide and ih/zoom high; x, y is its top-left corner.
       - zoom_in: z goes from 1 to ZOOM; window centred: x = (iw-iw/zoom)/2, same idea for y
       - zoom_out: the same, backwards
       - pan_right: z stays at ZOOM; x goes from 0 (left edge) to iw-iw/zoom (right edge); y centred
       - pan_left: the other way
       - ai_clip: same as zoom_in (no real clips yet)
    5. Encode: "-c:v", "libx264", "-pix_fmt", "yuv420p" (phones can't play what a PNG gives),
       and the output path last.

    The tests evaluate your expressions in Python, so use only numbers, + - * / ( ) and the
    names on, zoom, iw, ih (min and max are fine; ffmpeg's if() and ^ are not).
    """
    raise NotImplementedError("your turn: follow the hints above")
