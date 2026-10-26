"""images.py without fal, OpenAI or money: the placeholder for real, the OpenAI and fal paths with fakes."""

import base64
import io
from types import SimpleNamespace as NS

import httpx
import openai
import pytest
from PIL import Image

from shortsmith import cli, ledger, llm
from shortsmith.media import images


def test_mock_is_the_default_and_only_0_turns_it_off(monkeypatch):
    monkeypatch.delenv("MOCK_MEDIA", raising=False)
    assert images.mock_media() is True  # forgotten setting = free
    monkeypatch.setenv("MOCK_MEDIA", "0")
    assert images.mock_media() is False


def test_placeholder_is_a_vertical_full_hd_png(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MEDIA", "1")
    path = images.generate_image("a cozy cafe", tmp_path / "scenes" / "scene_01.png", label="Scene 1")
    with Image.open(path) as image:
        assert (image.size, image.format) == ((1080, 1920), "PNG")


def test_same_prompt_gives_the_same_placeholder_colour(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MEDIA", "1")
    first = images.generate_image("a cozy cafe", tmp_path / "a.png")
    second = images.generate_image("a cozy cafe", tmp_path / "b.png")
    with Image.open(first) as a, Image.open(second) as b:
        assert a.getpixel((5, 5)) == b.getpixel((5, 5))


def test_flux_path_queues_records_and_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MEDIA", "0")
    monkeypatch.setenv("IMAGE_PROVIDER", "fal")
    monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")
    sent = {}

    def fake_subscribe(model, arguments, **kwargs):
        sent.update(model=model, arguments=arguments)
        return {"images": [{"url": "https://fal.example/result.png"}]}

    monkeypatch.setattr(images.fal_client, "subscribe", fake_subscribe)
    monkeypatch.setattr(images, "_download", lambda url, path: path.write_bytes(b"fake png"))
    ledger_path = tmp_path / "ledger.jsonl"

    out = images.generate_image("a cozy cafe", tmp_path / "scene.png", ledger_path=ledger_path)

    assert out.read_bytes() == b"fake png"
    assert (sent["model"], sent["arguments"]["image_size"]) == ("fal-ai/flux/schnell", "portrait_16_9")
    assert ledger.total_cost(ledger_path) == pytest.approx(0.003)


# --- OpenAI's GPT Image (the OpenAI switch, 2026-10-26): fakes only, nothing is generated ---------------------


def png_b64(size: tuple[int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeImages:
    """Mimics client.images.generate(**kwargs): records the request, answers with a base64 PNG."""

    def __init__(self, size=(1024, 1536), usage=None, error: Exception | None = None):
        self.size, self.usage, self.error, self.requests = size, usage, error, []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return NS(data=[NS(b64_json=png_b64(self.size))], usage=self.usage)


def real_openai(monkeypatch, fake: FakeImages) -> None:
    monkeypatch.setenv("MOCK_MEDIA", "0")
    monkeypatch.setattr(llm, "_client", lambda *_: NS(images=fake))


def test_the_image_source_follows_mock_media_then_image_provider(monkeypatch):
    monkeypatch.delenv("MOCK_MEDIA", raising=False)
    assert images.source().provider == "mock" and images.price_per_image() == 0  # the default costs nothing
    real = images.source(real=True)
    assert (real.provider, real.model, real.quality, real.size, real.key_env) == (
        "openai", "gpt-image-2", "low", "1024x1536", "OPENAI_API_KEY")
    assert real.price_usd == pytest.approx(0.005)
    monkeypatch.setenv("OPENAI_IMAGE_SIZE", "1024x1024")  # fewer pixels, a smaller estimate
    assert images.source(real=True).price_usd == pytest.approx(0.005 * 1024 / 1536)
    monkeypatch.setenv("IMAGE_PROVIDER", "fal")
    assert (images.source(real=True).model, images.source(real=True).price_usd) == ("fal-ai/nano-banana", 0.0398)
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")
    assert images.source(real=True).provider == "mock"
    monkeypatch.setenv("IMAGE_PROVIDER", "dalle")
    with pytest.raises(ValueError, match="IMAGE_PROVIDER"):
        images.source(real=True)


def test_an_openai_image_is_asked_for_in_portrait_and_cropped_to_9_16(monkeypatch, tmp_path):
    fake = FakeImages(usage=NS(input_tokens=40, output_tokens=200, input_tokens_details=NS(text_tokens=40, image_tokens=0)))
    real_openai(monkeypatch, fake)
    out = images.generate_image("a cozy cafe", tmp_path / "scenes" / "scene_01.png", ledger_path=tmp_path / "ledger.jsonl")
    assert fake.requests == [{"model": "gpt-image-2", "prompt": "a cozy cafe", "size": "1024x1536", "quality": "low",
                              "n": 1, "output_format": "png"}]
    with Image.open(out) as image:
        assert (image.size, image.format) == ((864, 1536), "PNG")  # 9:16, the sides trimmed as the editor would


def test_an_openai_image_is_priced_from_its_usage_and_recorded(monkeypatch, tmp_path):
    usage = NS(input_tokens=40, output_tokens=200, input_tokens_details=NS(text_tokens=40, image_tokens=0))
    real_openai(monkeypatch, FakeImages(usage=usage))
    path = tmp_path / "ledger.jsonl"
    images.generate_image("a cozy cafe", tmp_path / "s.png", step="agent", ledger_path=path)
    entry = ledger.entries(path)[0]
    assert (entry.step, entry.model, entry.prompt_tokens, entry.completion_tokens) == ("agent", "gpt-image-2", 40, 200)
    assert entry.cost_usd == pytest.approx((40 * 5.00 + 200 * 30.00) / 1e6)


def test_without_usage_an_image_costs_the_tables_estimate(monkeypatch, tmp_path):
    real_openai(monkeypatch, FakeImages(usage=None))
    monkeypatch.setenv("OPENAI_IMAGE_QUALITY", "medium")
    images.generate_image("a cozy cafe", tmp_path / "s.png", ledger_path=tmp_path / "ledger.jsonl")
    assert ledger.total_cost(tmp_path / "ledger.jsonl") == pytest.approx(0.041)


def test_an_unverified_organization_gets_one_line_and_the_next_step(monkeypatch, tmp_path):
    request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
    refused = openai.PermissionDeniedError(
        "Error code: 403 - Your organization must be verified to use the model `gpt-image-2`. Please go to: "
        "https://platform.openai.com/settings/organization/general and click on Verify Organization.",
        response=httpx.Response(403, request=request), body=None)
    real_openai(monkeypatch, FakeImages(error=refused))
    with pytest.raises(images.VerificationRequired) as caught:
        images.generate_image("a cozy cafe", tmp_path / "s.png", ledger_path=tmp_path / "ledger.jsonl")
    message, hint, code = cli.explain(caught.value, None)
    assert message == "OpenAI refused gpt-image-2: the organization must be verified to use GPT Image models (403)"
    assert "Settings -> Organization -> General" in hint and code == cli.ExitCode.ERROR
    assert not (tmp_path / "ledger.jsonl").exists() and not (tmp_path / "s.png").exists()  # nothing made, nothing billed


def test_fit_portrait_keeps_9_16_and_trims_anything_else():
    assert images.fit_portrait(Image.new("RGB", (1080, 1920))).size == (1080, 1920)
    assert images.fit_portrait(Image.new("RGB", (1024, 1024))).size == (576, 1024)
    assert images.fit_portrait(Image.new("RGB", (864, 2000))).size == (864, 1536)


def test_nano_banana_on_fal_asks_for_9_16(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MEDIA", "0")
    monkeypatch.setenv("IMAGE_PROVIDER", "fal")
    sent = {}

    def fake_subscribe(model, arguments, **kwargs):
        sent.update(model=model, arguments=arguments)
        return {"images": [{"url": "https://fal.example/result.png"}]}

    monkeypatch.setattr(images.fal_client, "subscribe", fake_subscribe)
    monkeypatch.setattr(images, "_download", lambda url, path: path.write_bytes(b"fake png"))
    images.generate_image("a cozy cafe", tmp_path / "scene.png", ledger_path=tmp_path / "ledger.jsonl")
    assert sent["model"] == "fal-ai/nano-banana" and sent["arguments"]["aspect_ratio"] == "9:16"
    assert ledger.total_cost(tmp_path / "ledger.jsonl") == pytest.approx(0.0398)
