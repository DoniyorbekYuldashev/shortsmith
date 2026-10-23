"""Hero clips: one scene's image -> a 5-second AI video (image-to-video on fal.ai), or a free mock.

One clip per video, always after a person said yes (the graph's approval step, Step 6.6). Both
paths take the same arguments and write the same kind of file (an mp4 the editor fits to the
scene), so nothing downstream knows which one ran, exactly like images.py.

The real path has never been run: MOCK_MEDIA=0 plus a "yes" are both needed. Before the first
real clip, check CLIP_MODEL's live price and arguments on fal.ai (CLAUDE.md: the old plan figure of
$0.10-0.25 per clip was too optimistic).
"""

import subprocess
import tempfile
import urllib.request
from pathlib import Path

import fal_client
from PIL import Image, ImageDraw, ImageFont

from shortsmith import ledger
from shortsmith.media.images import mock_media

CLIP_MODEL = "fal-ai/kling-video/v2.1/standard/image-to-video"  # verify on fal.ai before the first real clip
CLIP_SECONDS = 5
CLIP_PRICE_USD = ledger.MEDIA_PRICES[CLIP_MODEL]  # the estimate the approval question shows
MOCK_SIZE = (540, 960)  # half size: a placeholder doesn't need full HD, the editor scales it up


def animate_image(image: Path, motion_prompt: str, out_path: Path, step: str = "producer",
                  ledger_path: Path = ledger.DEFAULT_PATH) -> Path:
    """Write a CLIP_SECONDS-long vertical clip of `image` moving as `motion_prompt` says."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mock_media():
        return _mock(image, out_path)
    return _fal(image, motion_prompt, out_path, step, ledger_path)


def mock_clip_cmd(image: Path, out: Path, seconds: int = CLIP_SECONDS) -> list[str]:
    """A clip that moves visibly unlike Ken Burns (the colours drift and the frame sways), so a
    mock hero scene is easy to spot in the final video. Free and offline."""
    w, h = MOCK_SIZE
    video_filter = ",".join([
        f"scale={w}:{h}:force_original_aspect_ratio=increase",
        f"crop={w}:{h}",
        "hue=h=40*t",  # colours rotate 40 degrees a second
        "rotate=0.04*sin(2*PI*t/2.5):fillcolor=black",  # a slow sway
    ])
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", image.as_posix(),
            "-vf", video_filter, "-t", str(seconds), "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", out.as_posix()]


def _mock(image: Path, out_path: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        labelled = Path(tmp) / "frame.png"
        card = Image.open(image).convert("RGB")
        ImageDraw.Draw(card).text((60, card.height - 260), "MOCK AI CLIP", fill="yellow",
                                  font=ImageFont.load_default(size=int(card.width / 9)))
        card.save(labelled)
        subprocess.run(mock_clip_cmd(labelled, out_path), capture_output=True, text=True, check=True)
    return out_path


def _fal(image: Path, motion_prompt: str, out_path: Path, step: str, ledger_path: Path) -> Path:
    """Upload the image, queue the job, wait, record the price, then download the clip."""
    price = ledger.MEDIA_PRICES[CLIP_MODEL]  # no price, no call
    image_url = fal_client.upload_file(image)
    result = fal_client.subscribe(CLIP_MODEL, arguments={
        "prompt": motion_prompt, "image_url": image_url, "duration": str(CLIP_SECONDS), "aspect_ratio": "9:16",
    })
    ledger.record(  # before the download: the clip is paid for even if downloading fails
        ledger.LedgerEntry(step=step, model=CLIP_MODEL, prompt_tokens=0, completion_tokens=0, cost_usd=price),
        ledger_path,
    )
    urllib.request.urlretrieve(result["video"]["url"], out_path)
    return out_path
