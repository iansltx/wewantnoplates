"""Central, editable configuration for the wewantnoplates pipeline.

Everything you most likely want to tweak lives in this one file. Command-line
flags can override most of these values per-invocation, but this is the single
place where defaults are defined.

The one hard requirement from the brief: the destination resolution is
expressed as a single "max pixels on the longest side" number, and it is
editable here — change `MAX_SIDE_PIXELS` (and only that) to resize output.
"""

# ---------------------------------------------------------------------------
# Destination resolution
# ---------------------------------------------------------------------------
# Maximum number of pixels along the *longest* side of the generated image.
# The shorter side is derived automatically to preserve the source aspect ratio
# (rounded to a multiple of 8, which the FLUX generator requires).
#
# Examples: 1024 -> square-ish 1024x1024; 896 -> max 896 on the longest side.
MAX_SIDE_PIXELS: int = 1024

# ---------------------------------------------------------------------------
# Cloud "understanding" step (via Ollama)
# ---------------------------------------------------------------------------
# Where Ollama is reachable. The understanding model is cloud-hosted in this
# setup (see `ollama list`), but Ollama itself still runs locally on this host.
OLLAMA_HOST: str = "http://localhost:11434"

# A *vision-capable* model for image understanding. This is the "cloud" model:
# in this project's setup it's `minimax-m3:cloud` (capabilities include `vision`).
# (Alternatives already installed with vision: kimi-k3:cloud, kimi-k2.7-code:cloud.)
VISION_MODEL: str = "minimax-m3:cloud"

# How long to allow the (remote, cloud) understanding call to run.
UNDERSTAND_TIMEOUT_SECONDS: float = 300.0

# ---------------------------------------------------------------------------
# Local "generation" step (runs on-device; fits comfortably in 64 GB VRAM)
# ---------------------------------------------------------------------------
# Backend used to render the transformed image. Currently supported: "diffusers"
# (image-to-image, runs on Apple Silicon via MPS or a CUDA GPU).
GENERATOR_BACKEND: str = "diffusers"

# Hugging Face model id for the image-to-image pipeline.
# SDXL is a good quality/speed default; FLUX.1-dev gives higher quality but is
# slower and needs more memory (fine in 64 GB).
IMG2IMG_MODEL: str = "stabilityai/stable-diffusion-xl-base-1.0"

# How strongly the edit is applied (0..1). Lower = closer to the source photo.
IMG2IMG_STRENGTH: float = 0.65

# Generation parameters.
GENERATOR_STEPS: int = 40
GENERATOR_GUIDANCE: float = 7.5

# Fixed seed for reproducible output. Set to None for random generations.
GENERATOR_SEED: int | None = 42

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# Where generated images (and sidecar .json of the prompt) are written.
OUTPUT_DIR: str = "output"
