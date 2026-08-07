"""End-to-end pipeline: input image + prompt -> transformed image.

   1. Load the input image (local file or URL).
   2. Ask a cloud vision model (via Ollama) for a *transformation prompt* that
      moves the food off the plate, r/wewantplates style. This is the sidecar
      prompt only — it is NOT an image description.
   3. Generate a new image locally with image-to-image diffusion: source photo
      + transformation prompt, at a size derived from the source aspect ratio
      and `config.MAX_SIDE_PIXELS`.
   4. Send the result back to the vision model to verify the transformation was
      applied; retry generation (with a fresh seed) if it wasn't.
   5. Save the image (plus a sidecar JSON recording the prompt and verdict) to
      the output directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config
from .generate import ImageGenerator, build_generator
from .io_util import (
    compute_target_size,
    image_to_bytes,
    load_image,
    make_output_path,
)
from .understand import Understanding, Verification, understand, verify


@dataclass
class PipelineResult:
    input_source: str
    transformation_prompt: str
    size: tuple[int, int]
    output_path: Path | None  # None when --dry-run
    prompt_path: Path | None
    verification: Verification | None = None  # None when --dry-run or --no-verify
    attempts: int = 0  # render attempts made (1 + retries when verification failed)


def run_pipeline(
    image_source: str,
    *,
    max_side: int | None = None,
    out_dir: str | None = None,
    vision_model: str | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    verify_output: bool = True,
    retries: int | None = None,
    strength: float | None = None,
    verify_model: str | None = None,
    generator: ImageGenerator | None = None,
) -> PipelineResult:
    """Execute the full pipeline and return the result.

    ``generator`` is injectable for testing; when omitted the configured backend
    (``config.GENERATOR_BACKEND``) is used.

    ``verify_output`` controls whether the generated image is sent back to the
    vision model for a pass/fail check; ``retries`` (default
    ``config.VERIFY_RETRIES``) is the number of extra render attempts made when
    the check fails. Each attempt uses a fresh seed. ``strength`` overrides
    ``config.IMG2IMG_STRENGTH``. ``verify_model`` overrides
    ``config.VERIFY_MODEL`` (the vision model used for the pass/fail check).
    """
    # 1. Ingest -----------------------------------------------------------------
    src = load_image(image_source)
    src_w, src_h = src.size

    # 2. Understand (cloud vision via Ollama) -> transformation prompt only -----
    image_bytes = image_to_bytes(src)
    understanding: Understanding = understand(image_bytes, vision_model=vision_model)

    # 3. Determine output size ------------------------------------------------
    target = compute_target_size(src_w, src_h, max_side=max_side)

    # 4. Generate (local image-to-image), verify, and retry --------------------
    output_path = None
    verification = None
    attempts = 0
    if not dry_run:
        gen = generator if generator is not None else build_generator(strength=strength)
        base_seed = seed if seed is not None else config.GENERATOR_SEED
        max_attempts = 1 + (retries if retries is not None else config.VERIFY_RETRIES)

        result_image = None
        for attempt in range(max_attempts):
            attempts = attempt + 1
            # Vary the seed on each retry so a failed render doesn't repeat.
            attempt_seed = None if base_seed is None else base_seed + attempt
            result_image = gen.generate(
                image=src,
                prompt=understanding.transformation_prompt,
                width=target[0],
                height=target[1],
                seed=attempt_seed,
            )
            if not verify_output:
                break
            verification = verify(
                image_to_bytes(result_image),
                image_bytes,
                understanding.transformation_prompt,
                verify_model=verify_model,
            )
            if verification.passed:
                break

        assert result_image is not None  # max_attempts >= 1
        output_path = make_output_path(out_dir=out_dir)
        result_image.save(output_path)

    # 5. Sidecar JSON ----------------------------------------------------------
    prompt_path = make_output_path(extension=".json", out_dir=out_dir)
    prompt_path.write_text(
        json.dumps(
            {
                "input": image_source,
                "max_side_pixels": max_side or config.MAX_SIDE_PIXELS,
                "output_size": list(target),
                "transformation_prompt": understanding.transformation_prompt,
                "verify_attempts": attempts,
                "verify_passed": verification.passed if verification else None,
                "verify_reason": verification.reason if verification else None,
            },
            indent=2,
        )
    )  # write_text returns char count; intentionally unused

    return PipelineResult(
        input_source=image_source,
        transformation_prompt=understanding.transformation_prompt,
        size=target,
        output_path=output_path,
        prompt_path=prompt_path,
        verification=verification,
        attempts=attempts,
    )
