"""Scene images: OpenAI's GPT Image, or fal.ai (Nano Banana, FLUX), or a free placeholder while MOCK_MEDIA is on.

Every path takes the same arguments and writes the same kind of file (a 9:16 PNG), so the rest of
the pipeline never needs to know which one ran.

Which one: MOCK_MEDIA=1 (the default) always makes placeholders. With real media (--real, after its
estimate and a yes), IMAGE_PROVIDER picks the source:
- openai (the default since 2026-10-26: paid from the OpenAI credit): OPENAI_IMAGE_MODEL (gpt-image-2),
  OPENAI_IMAGE_QUALITY (low / medium / high), OPENAI_IMAGE_SIZE (1024x1536, the portrait size every GPT
  Image model takes). 1024x1536 is 2:3, not 9:16, so the picture is centre-cropped to 864x1536 on save:
  the file on disk is what the video shows (the editor would cut the same strips off the sides).
- fal: FAL_IMAGE_MODEL, fal-ai/nano-banana ($0.0398, asked for 9:16) or fal-ai/flux/schnell (~$0.003).
- mock: placeholders even with --real (real hero clips from placeholder pictures, say).
"""

import argparse
import base64
import io
import os
import textwrap
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import fal_client
import openai
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from shortsmith import ledger, llm

FLUX_MODEL = "fal-ai/flux/schnell"
NANO_BANANA = "fal-ai/nano-banana"
FLUX_SIZE = "portrait_16_9"  # 576x1024: FLUX is priced by megapixels, and ffmpeg scales it up later
PLACEHOLDER_SIZE = (1080, 1920)  # full-HD vertical, the size of the final video
PORTRAIT = 9 / 16

IMAGE_PROVIDERS = ("openai", "fal", "mock")
DEFAULTS = {"IMAGE_PROVIDER": "openai", "OPENAI_IMAGE_MODEL": "gpt-image-2", "OPENAI_IMAGE_QUALITY": "low",
            "OPENAI_IMAGE_SIZE": "1024x1536", "FAL_IMAGE_MODEL": NANO_BANANA}
PRICED_SIZE = 1024 * 1536  # the size ledger.IMAGE_PRICES is for


def mock_media() -> bool:
    """Mock unless MOCK_MEDIA is explicitly "0": a forgotten setting must never cost money."""
    return os.environ.get("MOCK_MEDIA", "1") != "0"


def setting(name: str) -> str:
    return os.environ.get(name, "").strip() or DEFAULTS[name]


def image_provider() -> str:
    """Where REAL images come from (IMAGE_PROVIDER); MOCK_MEDIA=1 makes placeholders whatever this says."""
    name = setting("IMAGE_PROVIDER").lower()
    if name not in IMAGE_PROVIDERS:
        raise ValueError(f"IMAGE_PROVIDER={name!r}, expected one of {', '.join(IMAGE_PROVIDERS)}")
    return name


class ImageSource(BaseModel):
    """What an image would be made with, and its list price, before anything is made."""

    provider: str  # "openai", "fal" or "mock"
    model: str  # "gpt-image-2", "fal-ai/nano-banana", "placeholder"
    quality: str | None = None  # OpenAI only
    size: str | None = None  # OpenAI only
    price_usd: float  # per image, list price (an estimate for OpenAI: the real one comes from the response)
    key_env: str | None = None  # the key it needs

    @property
    def label(self) -> str:
        if self.provider == "mock":
            return "mock placeholders (free)"
        extra = f" ({self.quality}, {self.size})" if self.quality else ""
        return f"{self.provider} {self.model}{extra}, ~${self.price_usd:.4f} an image"


def openai_price(model: str, quality: str, size: str) -> float:
    """The per-image estimate: ledger.IMAGE_PRICES (for 1024x1536), scaled by pixel count for other sizes."""
    table = ledger.IMAGE_PRICES.get(ledger.base_model(model))
    if table is None or quality not in table:
        raise ValueError(f"No image price for {model!r} at quality {quality!r}: add it to IMAGE_PRICES in ledger.py")
    width, height = (int(n) for n in size.lower().split("x"))
    return table[quality] * width * height / PRICED_SIZE


def source(real: bool | None = None) -> ImageSource:
    """The image source for a run: real or not (default: what MOCK_MEDIA says), then IMAGE_PROVIDER."""
    real = not mock_media() if real is None else real
    name = image_provider() if real else "mock"
    if name == "openai":
        model, quality, size = setting("OPENAI_IMAGE_MODEL"), setting("OPENAI_IMAGE_QUALITY"), setting("OPENAI_IMAGE_SIZE")
        return ImageSource(provider=name, model=model, quality=quality, size=size,
                           price_usd=openai_price(model, quality, size), key_env="OPENAI_API_KEY")
    if name == "fal":
        model = setting("FAL_IMAGE_MODEL")
        if model not in ledger.MEDIA_PRICES:
            raise ValueError(f"No price for {model!r}: add it to MEDIA_PRICES in ledger.py")
        return ImageSource(provider=name, model=model, price_usd=ledger.MEDIA_PRICES[model], key_env="FAL_KEY")
    return ImageSource(provider="mock", model="placeholder", price_usd=0.0)


def price_per_image(real: bool | None = None) -> float:
    return source(real).price_usd


def generate_image(
    prompt: str,
    out_path: Path,
    label: str = "",
    step: str = "producer",
    ledger_path: Path = ledger.DEFAULT_PATH,
) -> Path:
    """Write one vertical image for `prompt` to `out_path` and return the path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen = source()
    if chosen.provider == "openai":
        return _openai(prompt, out_path, chosen, step, ledger_path)
    if chosen.provider == "fal":
        return _fal(prompt, out_path, chosen.model, step, ledger_path)
    return _placeholder(prompt, out_path, label)


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


# --- OpenAI -------------------------------------------------------------------------------------------


class VerificationRequired(RuntimeError):
    """OpenAI's 403 for GPT Image models: the organization isn't verified yet. Nothing was made or billed."""

    hint = ("verify your organization in the OpenAI dashboard: Settings -> Organization -> General -> Verify "
            "Organization (access can take a few minutes after that); or IMAGE_PROVIDER=fal")


def openai_request(prompt: str, src: ImageSource) -> dict[str, Any]:
    """The images.generate arguments. GPT Image models always answer with base64 (no URL to download)."""
    return {"model": src.model, "prompt": prompt, "size": src.size, "quality": src.quality, "n": 1,
            "output_format": "png"}


def usage_cost(model: str, usage: Any) -> float | None:
    """List price from the response's own token counts; None when it has none (then the table's estimate).
    Every output token is priced at the image-output rate, the dearest one."""
    price = ledger.IMAGE_TOKEN_PRICES.get(ledger.base_model(model))
    if price is None or usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    text_in = getattr(details, "text_tokens", None) if details else None
    image_in = (getattr(details, "image_tokens", 0) or 0) if details else 0
    if text_in is None:
        text_in = usage.input_tokens - image_in
    return (text_in * price["text_input"] + image_in * price["image_input"] + usage.output_tokens * price["output"]) / 1_000_000


def fit_portrait(image: Image.Image) -> Image.Image:
    """Centre-crop to 9:16 (1024x1536 -> 864x1536): cut what the editor's cover-and-crop would cut anyway."""
    width, height = image.size
    if abs(width / height - PORTRAIT) < 0.005:
        return image
    if width / height > PORTRAIT:  # too wide: trim the sides
        new_width = round(height * PORTRAIT / 2) * 2
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    new_height = round(width / PORTRAIT / 2) * 2  # too tall: trim top and bottom
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def _openai(prompt: str, out_path: Path, src: ImageSource, step: str, ledger_path: Path) -> Path:
    """One image from OpenAI: priced from its usage, recorded, then cropped to 9:16 and saved."""
    try:
        response = llm._client("openai").images.generate(**openai_request(prompt, src))
    except openai.PermissionDeniedError as e:  # 403: nothing generated, nothing billed
        if "verif" in str(e).lower():
            raise VerificationRequired(f"OpenAI refused {src.model}: the organization must be verified to use GPT "
                                       "Image models (403)") from e
        raise
    except openai.RateLimitError as e:
        if (no_credit := llm._no_credit(e)) is not None:
            raise no_credit from e
        raise
    usage = getattr(response, "usage", None)
    cost = usage_cost(src.model, usage)
    ledger.record(  # before the picture is decoded or saved: it is paid for even if that fails
        ledger.LedgerEntry(step=step, model=src.model, prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                           completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                           cost_usd=cost if cost is not None else src.price_usd),
        ledger_path,
    )
    with Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json))) as image:
        fit_portrait(image.convert("RGB")).save(out_path, format="PNG")
    return out_path


# --- fal.ai -----------------------------------------------------------------------------------------------


def fal_arguments(prompt: str, model: str) -> dict[str, Any]:
    """Each fal model names its size its own way: FLUX a preset, Nano Banana an aspect ratio."""
    if model == FLUX_MODEL:
        return {"prompt": prompt, "image_size": FLUX_SIZE, "num_images": 1, "output_format": "png"}
    return {"prompt": prompt, "aspect_ratio": "9:16", "num_images": 1, "output_format": "png"}


def _fal(prompt: str, out_path: Path, model: str, step: str, ledger_path: Path) -> Path:
    """Queue a job on fal, wait for it, record the price, then download the image."""
    price = ledger.MEDIA_PRICES[model]  # fail before paying if the model has no price
    result = fal_client.subscribe(model, arguments=fal_arguments(prompt, model))  # submits, then polls until done
    ledger.record(  # recorded before the download: the image is paid for even if downloading fails
        ledger.LedgerEntry(step=step, model=model, prompt_tokens=0, completion_tokens=0, cost_usd=price),
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
    print(f"\n{source().label} | script by {result.model} | ledger total ${ledger.total_cost(ledger.DEFAULT_PATH):.6f}")
