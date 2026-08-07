"""Tests for the pipeline orchestration (generation stubbed out)."""

from unittest.mock import patch

from PIL import Image

from wewantnoplates.pipeline import run_pipeline
from wewantnoplates.understand import Verification


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


def _mock_verification(passed=True, reason="ok"):
    return Verification(passed=passed, reason=reason, raw_response="{}")


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
@patch("wewantnoplates.pipeline.verify")
def test_run_pipeline_saves_image_and_prompt(mock_verify, mock_understand, mock_load, tmp_path):
    source = Image.new("RGB", (960, 720), (10, 200, 30))
    mock_load.return_value = source
    mock_understand.return_value = _mock_understanding()
    mock_verify.return_value = _mock_verification(passed=True)

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
    assert result.attempts == 1
    assert result.verification is not None and result.verification.passed

    saved = Image.open(result.output_path)
    assert saved.size == (1024, 768)
    # Source pixel reached the generator (top-left corner preserved).
    assert saved.getpixel((0, 0)) == (10, 200, 30)


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
@patch("wewantnoplates.pipeline.verify")
def test_run_pipeline_dry_run_skips_generation(mock_verify, mock_understand, mock_load, tmp_path):
    mock_load.return_value = Image.new("RGB", (960, 720), (10, 10, 10))
    mock_understand.return_value = _mock_understanding()

    result = run_pipeline("unused.jpg", out_dir=str(tmp_path), dry_run=True)

    assert result.output_path is None  # no image written
    assert result.prompt_path.exists()  # prompt sidecar still written
    mock_verify.assert_not_called()


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
@patch("wewantnoplates.pipeline.verify")
def test_prompt_sidecar_has_no_description(mock_verify, mock_understand, mock_load, tmp_path):
    mock_load.return_value = Image.new("RGB", (10, 10), (0, 0, 0))
    mock_understand.return_value = _mock_understanding(transformation_prompt="x")
    mock_verify.return_value = _mock_verification(passed=True)

    result = run_pipeline("unused.jpg", out_dir=str(tmp_path), dry_run=True)
    import json

    payload = json.loads(result.prompt_path.read_text())
    assert "transformation_prompt" in payload
    assert "description" not in payload


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
@patch("wewantnoplates.pipeline.verify")
def test_verify_failure_retries_generation(mock_verify, mock_understand, mock_load, tmp_path):
    mock_load.return_value = Image.new("RGB", (960, 720), (10, 10, 10))
    mock_understand.return_value = _mock_understanding()
    # Fail once, pass on the retry.
    mock_verify.side_effect = [
        _mock_verification(passed=False, reason="plate still visible"),
        _mock_verification(passed=True, reason="moved to a hubcap"),
    ]

    result = run_pipeline("unused.jpg", out_dir=str(tmp_path), generator=_FakeGenerator())

    assert result.attempts == 2
    assert mock_verify.call_count == 2
    assert result.verification is not None and result.verification.passed


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
@patch("wewantnoplates.pipeline.verify")
def test_verify_exhausts_retries_and_reports_failure(
    mock_verify, mock_understand, mock_load, tmp_path
):
    mock_load.return_value = Image.new("RGB", (960, 720), (10, 10, 10))
    mock_understand.return_value = _mock_understanding()
    mock_verify.return_value = _mock_verification(passed=False, reason="nope")

    # 1 render + 3 retries = 4 attempts.
    result = run_pipeline(
        "unused.jpg", out_dir=str(tmp_path), generator=_FakeGenerator(), retries=3
    )

    assert result.attempts == 4
    assert mock_verify.call_count == 4
    assert result.verification is not None and not result.verification.passed
    # Sidecar records the failure verdict.
    import json

    payload = json.loads(result.prompt_path.read_text())
    assert payload["verify_passed"] is False
    assert payload["verify_attempts"] == 4


@patch("wewantnoplates.pipeline.load_image")
@patch("wewantnoplates.pipeline.understand")
@patch("wewantnoplates.pipeline.verify")
def test_no_verify_generates_single_attempt(mock_verify, mock_understand, mock_load, tmp_path):
    mock_load.return_value = Image.new("RGB", (960, 720), (10, 10, 10))
    mock_understand.return_value = _mock_understanding()

    result = run_pipeline(
        "unused.jpg",
        out_dir=str(tmp_path),
        generator=_FakeGenerator(),
        verify_output=False,
    )

    assert result.attempts == 1
    assert result.verification is None
    mock_verify.assert_not_called()
