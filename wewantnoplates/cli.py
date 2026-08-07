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
    help="Cloud vision model via Ollama (default: config.VISION_MODEL).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Generation seed for reproducibility (default: config.GENERATOR_SEED).",
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
    seed: int | None,
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
        seed=seed,
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


def entrypoint() -> None:  # console-script hook
    main(standalone_mode=True)
