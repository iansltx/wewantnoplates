"""Local "generation" step.

Renders the transformed image on-device with **image-to-image** diffusion: it
takes the *original* photo plus a short transformation prompt and produces the
edited image (food moved off the plate).

The default backend uses `diffusers`, which supports img2img and runs on Apple
Silicon via MPS or on a CUDA GPU (comfortably within a 64 GB machine). Only the
selected backend is imported at call time so the pipeline can still run in
`--dry-run` (prompt generation only) without the heavy ML deps installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from . import config


class ImageGenerator(ABC):
    """Generate an image from an input image + a transformation prompt."""

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        width: int,
        height: int,
        seed: int | None,
    ) -> Image.Image:
        ...


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DiffusersImg2ImgGenerator(ImageGenerator):
    """Image-to-image generation via the diffusers library."""

    name: str = "diffusers"

    def __init__(
        self,
        model_id: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        strength: float | None = None,
    ) -> None:
        self.model_id = model_id or config.IMG2IMG_MODEL
        self.steps = steps if steps is not None else config.GENERATOR_STEPS
        self.guidance = guidance if guidance is not None else config.GENERATOR_GUIDANCE
        self.strength = strength if strength is not None else config.IMG2IMG_STRENGTH

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        width: int,
        height: int,
        seed: int | None,
    ) -> Image.Image:
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:  # pragma: no cover - heavy dep
            raise SystemExit(
                "The `diffusers`/`torch` packages are required for generation.\n"
                "Install with:  pip install -e '.[gen]'"
            ) from exc

        device = _pick_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except TypeError:  # some model ids don't accept use_safetensors
            pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id, torch_dtype=dtype
            )
        pipe = pipe.to(device)

        init = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            image=init,
            strength=self.strength,
            guidance_scale=self.guidance,
            num_inference_steps=self.steps,
            generator=gen,
        ).images[0]
        return result


def build_generator(backend: str | None = None) -> ImageGenerator:
    """Return the configured generator backend instance."""
    backend = (backend or config.GENERATOR_BACKEND).lower()
    if backend in ("diffusers", "diffuser"):
        return DiffusersImg2ImgGenerator()
    raise SystemExit(f"Unknown GENERATOR_BACKEND: {backend!r}")
