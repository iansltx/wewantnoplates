"""Tests for the pipeline orchestration (generation stubbed out)."""

from unittest.mock import patch

from PIL import Image

from wewantnoplates.pipeline import run_pipeline


class _FakeGenerator:
    """Deterministic stand-in for the image-to-image generator."""

    def generate(self, image, prompt, width, height, seed):
        assert isinstance(image, Image.Image)
        assert isinstance(prompt, str) and prompt
        rendered = Image.new("RGB", (width, height), (200, 60, 60))
        # Prove the source image actually reaches the generator.
        rendered.putpixel((0, 0), image.convert("RGB").getpixel((0, 0)))
        return rendered


def _mock_understanding(transformation_prompt="pizza on a hubcap"):
    return type(
        "Understanding",
        (),
        {"transformation_prompt": transformation_prompt, "raw_response": "{}"},
    )()


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
def test_run_pipeline_saves_image_and_prompt(mock_understand, mock_load, tmp_path):
    source = Image.new("RGB", (960, 720), (10, 200, 30))
    mock_load.return_value = source
    mock_understand.return_value = _mock_understanding()

    result = run_pipeline(
        "unused.jpg",
        max_side=1024,
        out_dir=str(tmp_path),
        generator=_FakeGenerator(),
    )

    assert result.output_path is not None
    assert result.output_path.exists()
    assert result.prompt_path.exists()
    assert result.size == (1024, 768)

    saved = Image.open(result.output_path)
    assert saved.size == (1024, 768)
    # Source pixel reached the generator (top-left corner preserved).
    assert saved.getpixel((0, 0)) == (10, 200, 30)


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
def test_run_pipeline_dry_run_skips_generation(mock_understand, mock_load, tmp_path):
    mock_load.return_value = Image.new("RGB", (960, 720), (10, 10, 10))
    mock_understand.return_value = _mock_understanding()

    result = run_pipeline("unused.jpg", out_dir=str(tmp_path), dry_run=True)

    assert result.output_path is None  # no image written
    assert result.prompt_path.exists()  # prompt sidecar still written


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
def test_prompt_sidecar_has_no_description(mock_understand, mock_load, tmp_path):
    mock_load.return_value = Image.new("RGB", (10, 10), (0, 0, 0))
    mock_understand.return_value = _mock_understanding(transformation_prompt="x")

    result = run_pipeline("unused.jpg", out_dir=str(tmp_path), dry_run=True)
    import json

    payload = json.loads(result.prompt_path.read_text())
    assert "transformation_prompt" in payload
    assert "description" not in payload
