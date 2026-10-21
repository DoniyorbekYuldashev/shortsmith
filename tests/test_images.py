"""images.py without fal or money: the placeholder for real, the FLUX path with fakes."""

import pytest
from PIL import Image

from shortsmith import ledger
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
