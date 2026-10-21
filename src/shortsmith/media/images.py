"""Scene images: FLUX schnell on fal.ai, or a free placeholder while MOCK_MEDIA is on.

Both paths take the same arguments and write the same kind of file, so the rest of the
pipeline never needs to know which one ran.
"""

import argparse
import os
import textwrap
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

import fal_client
from PIL import Image, ImageDraw, ImageFont

from shortsmith import ledger

FLUX_MODEL = "fal-ai/flux/schnell"
FLUX_SIZE = "portrait_16_9"  # 576x1024: FLUX is priced by megapixels, and ffmpeg scales it up later
PLACEHOLDER_SIZE = (1080, 1920)  # full-HD vertical, the size of the final video


def mock_media() -> bool:
    """Mock unless MOCK_MEDIA is explicitly "0": a forgotten setting must never cost money."""
    return os.environ.get("MOCK_MEDIA", "1") != "0"


def generate_image(
    prompt: str,
    out_path: Path,
    label: str = "",
    step: str = "producer",
    ledger_path: Path = ledger.DEFAULT_PATH,
) -> Path:
    """Write one vertical image for `prompt` to `out_path` and return the path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mock_media():
        return _placeholder(prompt, out_path, label)
    return _flux(prompt, out_path, step, ledger_path)


def _placeholder(prompt: str, out_path: Path, label: str) -> Path:
    """A coloured card showing the label and prompt: free, instant, and easy to tell apart."""
    seed = zlib.crc32(prompt.encode("utf-8"))  # same prompt -> same colour, run after run
    colour = tuple(60 + (seed >> shift) % 140 for shift in (16, 8, 0))  # mid-tones keep white text readable
    image = Image.new("RGB", PLACEHOLDER_SIZE, colour)
    text = (f"{label}\n\n" if label else "") + "\n".join(textwrap.wrap(prompt, width=30))
    ImageDraw.Draw(image).multiline_text(
        (80, 220), text, fill="white", font=ImageFont.load_default(size=52), spacing=18
    )
    image.save(out_path)
    return out_path


def _flux(prompt: str, out_path: Path, step: str, ledger_path: Path) -> Path:
    """Queue a job on fal, wait for it, record the price, then download the image."""
    price = ledger.MEDIA_PRICES[FLUX_MODEL]  # fail before paying if the model has no price
    result = fal_client.subscribe(  # submits to fal's queue, then polls it until the job is done
        FLUX_MODEL,
        arguments={"prompt": prompt, "image_size": FLUX_SIZE, "num_images": 1, "output_format": "png"},
    )
    ledger.record(  # recorded before the download: the image is paid for even if downloading fails
        ledger.LedgerEntry(step=step, model=FLUX_MODEL, prompt_tokens=0, completion_tokens=0, cost_usd=price),
        ledger_path,
    )
    _download(result["images"][0]["url"], out_path)
    return out_path


def _download(url: str, out_path: Path) -> None:
    urllib.request.urlretrieve(url, out_path)


if __name__ == "__main__":
    from shortsmith.llm import write_script  # demo only: images.py itself doesn't need the LLM

    parser = argparse.ArgumentParser(description="Topic -> Script -> one image per scene.")
    parser.add_argument("topic")
    parser.add_argument("--lang", choices=["en", "uz", "ru"], default="en")
    args = parser.parse_args()

    script, result = write_script(args.topic, args.lang)
    run_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    for scene in script.scenes:
        path = generate_image(
            scene.visual_prompt, run_dir / "scenes" / f"scene_{scene.idx:02d}.png", f"Scene {scene.idx} - {scene.motion}"
        )
        print(f"scene {scene.idx}: {path}")
    print(f"\n{'MOCK placeholders' if mock_media() else 'FLUX images'} | script by {result.model} | "
          f"ledger total ${ledger.total_cost(ledger.DEFAULT_PATH):.6f}")
