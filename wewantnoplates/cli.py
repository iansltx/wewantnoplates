"""Command-line interface for the wewantnoplates pipeline."""

from __future__ import annotations

import click

from . import __version__
from .pipeline import run_pipeline


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("image", metavar="IMAGE", required=True)
@click.option(
    "--max-side",
    type=int,
    default=None,
    help="Max pixels on the longest side of the output (default: config.MAX_SIDE_PIXELS).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Directory for generated images (default: config.OUTPUT_DIR).",
)
@click.option(
    "--vision-model",
    default=None,
    help="Cloud vision model that writes the prompt (default: config.VISION_MODEL).",
)
@click.option(
    "--verify-model",
    default=None,
    help="Vision model used to verify the result (default: config.VERIFY_MODEL).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Generation seed for reproducibility (default: config.GENERATOR_SEED).",
)
@click.option(
    "--retries",
    type=int,
    default=None,
    help="Max regeneration retries when verification fails (default: config.VERIFY_RETRIES).",
)
@click.option(
    "--strength",
    type=float,
    default=None,
    help="IMG2IMG strength 0..1 (default: config.IMG2IMG_STRENGTH).",
)
@click.option(
    "--no-verify",
    is_flag=True,
    help="Skip verification/retry; render a single image.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run understanding + prompt generation only; do NOT render an image.",
)
@click.version_option(__version__, prog_name="wewantnoplates")
def main(
    image: str,
    max_side: int | None,
    out_dir: str | None,
    vision_model: str | None,
    verify_model: str | None,
    seed: int | None,
    retries: int | None,
    strength: float | None,
    no_verify: bool,
    dry_run: bool,
) -> None:
    """Turn food on a plate into food NOT on a plate (r/wewantplates style).

    IMAGE is a local file path or an http(s) URL.
    """
    result = run_pipeline(
        image,
        max_side=max_side,
        out_dir=out_dir,
        vision_model=vision_model,
        verify_model=verify_model,
        seed=seed,
        retries=retries,
        strength=strength,
        verify_output=not no_verify,
        dry_run=dry_run,
    )

    click.echo("Transformation prompt:")
    click.echo(f"  {result.transformation_prompt}")
    click.echo(f"\nOutput size: {result.size[0]}x{result.size[1]}")
    click.echo(f"Prompt saved: {result.prompt_path}")

    if dry_run:
        click.echo("(dry-run: image generation skipped)")
    else:
        click.echo(f"Image saved: {result.output_path}")
        _echo_verification(result, retries)


def _echo_verification(result, retries: int | None) -> None:
    """Print the verification verdict and how many attempts were made."""
    from . import config

    max_attempts = 1 + (retries if retries is not None else config.VERIFY_RETRIES)
    v = result.verification
    if v is None:
        click.echo("Verification: skipped")
    elif v.passed:
        click.echo(f"Verification: passed (attempt {result.attempts}/{max_attempts})")
    else:
        click.echo(
            f"Verification: FAILED after {result.attempts} attempt(s) — {v.reason}"
        )


def entrypoint() -> None:  # console-script hook
    main(standalone_mode=True)