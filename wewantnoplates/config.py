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

# Vision model used to VERIFY the generated image. A stronger model here gives
# more trustworthy pass/fail verdicts; a weak verifier can misread an absurd
# surface (e.g. a manhole cover) as a plate. Defaults to a stronger model than
# the understanding step. Override per-run with --verify-model.
VERIFY_MODEL: str = "kimi-k2.6:cloud"

# How long to allow the (remote, cloud) understanding call to run.
UNDERSTAND_TIMEOUT_SECONDS: float = 300.0

# ---------------------------------------------------------------------------
# Local "generation" step (runs on-device; fits comfortably in 64 GB VRAM)
# ---------------------------------------------------------------------------
# Backend used to render the transformed image. Currently supported: "diffusers"
# (image-to-image, runs on Apple Silicon via MPS or a CUDA GPU).
GENERATOR_BACKEND: str = "diffusers"

# Hugging Face model id for the image-to-image pipeline.
# FLUX.1-schnell (via a non-gated mirror; schnell is Apache-2.0) follows prompts
# far better than SDXL and renders in a few steps, which suits the verify/retry
# loop. The official black-forest-labs/FLUX.1-schnell repo works too once you
# accept its license on the HF page. SDXL is a lighter fallback but clings to
# the source composition.
IMG2IMG_MODEL: str = "unsloth/FLUX.1-schnell"

# How strongly the edit is applied (0..1). Lower = closer to the source photo.
# Set high enough that the plate actually disappears; verification+retries then
# catch any residual failures.
IMG2IMG_STRENGTH: float = 0.9

# Generation parameters. FLUX.1-schnell is a 4-step distilled model: it uses few
# steps and no classifier-free guidance (guidance 0). SDXL would want ~40 / 7.5.
GENERATOR_STEPS: int = 8
GENERATOR_GUIDANCE: float = 0.0

# Fixed seed for reproducible output. Set to None for random generations.
GENERATOR_SEED: int | None = 42

# How many times to regenerate if verification reports the transformation wasn't
# applied (e.g. the plate is still visible). The first render plus these retries
# are attempted, each with a fresh seed. Set to 0 for a single attempt.
VERIFY_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# Where generated images (and sidecar .json of the prompt) are written.
OUTPUT_DIR: str = "output"
