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
from typing import Any

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
    """Image-to-image generation via the diffusers library.

    The heavy diffusion pipeline is loaded lazily on the first ``generate`` call
    and cached on the instance, so retries (verification failures) reuse the
    already-loaded model instead of reloading it each time.
    """

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
        self._pipe: Any = None  # lazily loaded and reused across generate()
        self._device: str | None = None

    def _load_pipe(self):
        """Load the diffusion pipeline once and return the cached instance."""
        if self._pipe is not None:
            return self._pipe

        try:
            import torch
            from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
        except ImportError as exc:  # pragma: no cover - heavy dep
            raise SystemExit(
                "The `diffusers`/`torch` packages are required for generation.\n"
                "Install with:  pip install -e '.[gen]'"
            ) from exc

        self._device = _pick_device()
        # fp16 on CUDA and Apple Silicon so FLUX fits in 64 GB unified memory.
        dtype = torch.float16 if self._device != "cpu" else torch.float32
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
        self._pipe = pipe.to(self._device)
        return self._pipe

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        width: int,
        height: int,
        seed: int | None,
    ) -> Image.Image:
        import torch

        pipe = self._load_pipe()
        device = self._device or _pick_device()

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


def build_generator(backend: str | None = None, strength: float | None = None) -> ImageGenerator:
    """Return the configured generator backend instance.

    ``strength`` overrides ``config.IMG2IMG_STRENGTH`` when provided.
    """
    backend = (backend or config.GENERATOR_BACKEND).lower()
    if backend in ("diffusers", "diffuser"):
        return DiffusersImg2ImgGenerator(strength=strength)
    raise SystemExit(f"Unknown GENERATOR_BACKEND: {backend!r}")
