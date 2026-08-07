# wewantnoplates

Turn a photo of food *on a plate* into a photo of the same food defiantly
**NOT on a plate** — in the proud tradition of
[/r/wewantplates](https://www.reddit.com/r/wewantplates/). Give it a pizza on
ceramic; get back the same pizza balanced on a rusty garden shovel, a
circular-saw blade, or a dirty hubcap.

It's an **image-to-image** CLI pipeline. The generator's two inputs are the
**original photo** and a short **transformation prompt**; the output is the
edited image. A local vision model (gemma3:12b via Ollama) writes that
transformation prompt — it never describes the food from scratch (the generator
already sees the image). After rendering, a vision model **verifies** the
result: if the plate is still visible, the pipeline regenerates with a fresh
seed, up to a configurable number of retries. The whole pipeline runs locally
(generation + both vision steps) — nothing leaves the machine.

This project was partially intended as a test bench for how hard I could push
DeepSeek V4 Flash 0731, as well as whether I could make this work with fully
local models on my M1 Max MacBook Pro with 64GB of unified memory. All of the
code and most of the readme (see commits for exceptions; you can probably tell
which ones I wrote by hand) are generated, primarily by Flash but with a detour
to GLM 5.2 when I thought I could use it in the loop to verify images (nope).
Used Zed's built-in harness throughout, and Ollama Cloud for all cloud inference.

```
  ┌──────────────┐
  │ input image  │──────────────┐
  │ (file / URL) │              │
  └──────────────┘              ▼
         │            ┌──────────────────────────┐
         │            │ vision (Ollama, gemma3)  │
         │            │ writes transformation    │──▶ sidecar prompt
         │            │ prompt (img2img)         │
         │            └──────────────────────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
        ┌──────────────────────────┐
        │ local image-to-image     │──▶ transformed image
        │ diffusion (FLUX schnell) │    │
        └──────────────────────────┘    │
                      │                ▼
                      │      ┌──────────────────────────┐
                      │      │ vision (Ollama, gemma3)  │
                      │      │ verifies plate is gone   │── no → retry (fresh seed)
                      │      └──────────────────────────┘
                      ▼                         ── yes
               saved locally            saved locally
```

---

## Quickstart

```bash
pip install -e "./[gen]"
wewantnoplates photo_of_food_on_a_plate.jpg
# -> output/wewantnoplates-<timestamp>.png  (plus a sidecar .json)
```

The output image, the sidecar prompt, and the resolution are all printed to the
terminal. See [Installation](#installation) for prerequisites and
[Usage](#usage) for the full options.

---

## Requirements

- **Ollama** running (defaults to `http://localhost:11434`) with a
  **vision-capable** model. Both vision steps use the local `gemma3:12b` (multimodal).
  `gemma4:*` MLX builds are **text-only** (no `vision` capability) — they can't see
  images, so avoid them for vision steps. Cloud vision models (`minimax-m3:cloud`,
  `kimi-k2.6:cloud`) work too but are optional and billed; `glm-5.2:cloud` does
  **not** support image input via Ollama.
- **Python 3.10+** for the understanding path. The generation step additionally
  needs the `diffusers`/`torch` stack, which works on **Apple Silicon (MPS)** or
  a **CUDA GPU** (comfortably within a 64 GB machine; FLUX.1-schnell runs in fp16).
- First generation run downloads the model weights from Hugging Face. The default
  `IMG2IMG_MODEL` is FLUX.1-schnell (~34 GB download / ~35 GB on disk in fp16). A
  lighter fallback is SDXL (`stabilityai/stable-diffusion-xl-base-1.0`, ~7 GB
  download / ~14 GB on disk), but it clings to the source plate.
- FLUX.1-schnell's official repo is gated; the default uses a non-gated mirror
  (`unsloth/FLUX.1-schnell`). To use the official repo instead, accept its license
  on the HF page and set `IMG2IMG_MODEL = "black-forest-labs/FLUX.1-schnell"`.

---

## Installation

```bash
# 1. Virtualenv (3.10+; 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the package, including the heavy generation extra
pip install -e ".[gen]"
```

> On Apple Silicon you may prefer to pre-install the arm64 PyTorch wheel first
> for best performance; the `[gen]` extra's `torch` should otherwise resolve a
> working wheel.

---

## Usage

```
wewantnoplates [OPTIONS] IMAGE

  IMAGE is a local file path or an http(s) URL.
```

### Examples

```bash
# Local file
wewantnoplates my_steak_dinner.jpg

# Remote image
wewantnoplates https://example.com/pizza_on_a_plate.jpg

# Control resolution: max 768 px on the longest side
wewantnoplates my_steak_dinner.jpg --max-side 768

# Write somewhere specific + reproducible seed
wewantnoplates my_steak_dinner.jpg --out-dir ./gallery --seed 7

# Use a specific cloud vision model
wewantnoplates my_steak_dinner.jpg --vision-model minimax-m3:cloud

# Understanding + prompt only (no rendering) — handy for testing
wewantnoplates my_steak_dinner.jpg --dry-run

# Tune verification: allow up to 5 retries if the plate isn't gone
wewantnoplates my_steak_dinner.jpg --retries 5

# Sweep the edit strength without editing config.py (0..1)
wewantnoplates my_steak_dinner.jpg --strength 0.9

# Turn verification/retry off entirely (single render)
wewantnoplates my_steak_dinner.jpg --no-verify
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--max-side N` | `config.MAX_SIDE_PIXELS` (`1024`) | Max pixels along the longest output side |
| `--out-dir DIR` | `config.OUTPUT_DIR` (`output`) | Where images + sidecar JSON are written |
| `--vision-model MODEL` | `config.VISION_MODEL` | Cloud vision model (via Ollama) that writes the prompt |
| `--verify-model MODEL` | `config.VERIFY_MODEL` | Cloud vision model that judges pass/fail |
| `--seed N` | `config.GENERATOR_SEED` (`42`) | Generation seed for reproducibility |
| `--strength X` | `config.IMG2IMG_STRENGTH` (`0.9`) | Edit strength 0..1; higher = more transformation |
| `--retries N` | `config.VERIFY_RETRIES` (`3`) | Max regeneration retries when verification fails |
| `--no-verify` | off | Skip verification/retry; render a single image |
| `--dry-run` | off | Run understanding + prompt only; skip rendering |
| `-h` / `--help` | — | Show help and exit |
| `--version` | — | Show version and exit |

Typical output:

```
Transformation prompt:
  Lift the pizza off the plate and re-serve it on a rusty vintage chrome hubcap
  sitting on a dusty concrete garage floor, crumbs and grease smudged around the
  rim, warm overhead tungsten light, photorealistic, "fresh from the wheel."

Output size: 1024x768
Prompt saved: output/wewantnoplates-<timestamp>.json
Image saved:  output/wewantnoplates-<timestamp>.png
Verification: passed (attempt 1/4)
```

---

## How the two inputs combine

The generator is image-to-image:

- **Input image** — the original photo, resized to the target size.
- **Transformation prompt** — a short instruction (≤ ~60 words) that only
  states the *change*: take the food off the plate and re-serve it directly on
  a specific absurd surface in r/wewantplates style, and make the plate/tableware
  disappear. It does **not** describe the food, ingredients, or existing scene,
  and it deliberately does **not** ask the generator to preserve the original
  plate or table.

The separate vision model (gemma3 12B by default) looks at the photo and writes that
transformation prompt. That prompt is saved verbatim as a sidecar `.json` next
to the output image, so every edit is reproducible and inspectable.

The sidecar also records the **verification verdict**: how many render attempts
were made, whether the check passed, and the reason the model gave. If the check
failed, generation is retried with a fresh seed (see [Verification & retries](#verification--retries)).

---

## Configuration

All defaults live in **one file**: [`wewantnoplates/config.py`](wewantnoplates/config.py).

The single knob for resolution is:

```python
# Maximum number of pixels along the LONGEST side of the generated image.
# The shorter side is derived from the source aspect ratio.
MAX_SIDE_PIXELS: int = 1024
```

Change that value (and only that) to resize every future output. It's also
overridable per-invocation with `--max-side`.

Other notable settings in the same file:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama listens |
| `UNDERSTAND_TIMEOUT_SECONDS` | `300.0` | Max time to wait for the remote vision call |
| `GENERATOR_BACKEND` | `diffusers` | Image-to-image backend used for rendering |
| `VISION_MODEL` | `gemma3:12b` | Local model that writes the transformation prompt |
| `VERIFY_MODEL` | `gemma3:12b` | Local vision model that judges the pass/fail verdict |
| `IMG2IMG_MODEL` | `unsloth/FLUX.1-schnell` | Local image-to-image model |
| `IMG2IMG_STRENGTH` | `0.9` | How strongly the edit is applied (0..1); override per-run with `--strength` |
| `GENERATOR_STEPS` / `GENERATOR_GUIDANCE` | `8` / `0.0` | Diffusion sampling params (FLUX.1-schnell is distilled, no CFG) |
| `GENERATOR_SEED` | `42` | Reproducibility (set `None` for random) |
| `VERIFY_RETRIES` | `3` | Max regeneration retries if verification fails (each uses a fresh seed) |
| `OUTPUT_DIR` | `output` | Where images + sidecar JSON go |

### How the destination size works

The output preserves the input's **aspect ratio**. The longest side is capped at
`MAX_SIDE_PIXELS` and the shorter side is scaled to match, then rounded down to
a multiple of 8 (required by diffusion models). Examples from a 4:3 source:

| `MAX_SIDE_PIXELS` | Output |
| --- | --- |
| 1024 | 1024×768 |
| 768  | 768×576 |
| 512  | 512×384 |

---

## The three model roles

The pipeline splits work across two **local** vision roles and a **local**
generation role — everything runs on this machine:

- **Understanding — local.** `VISION_MODEL` (`gemma3:12b`) runs through Ollama
  locally. It looks at the photo and writes the transformation prompt.
  Vision-capable cloud models (`minimax-m3:cloud`, `kimi-k2.6:cloud`) can be
  swapped in, but they're optional and billed.
- **Generation — local.** `IMG2IMG_MODEL` (`unsloth/FLUX.1-schnell`, a non-gated
  mirror of the Apache-2.0 schnell) runs entirely on-device via `diffusers`/
  PyTorch (MPS on Apple Silicon, or CUDA) in fp16. The official
  `black-forest-labs/FLUX.1-schnell` repo works too once you accept its license on
  the HF page. SDXL (`stabilityai/stable-diffusion-xl-base-1.0`) is a lighter
  fallback but clings to the source composition and tends to keep the plate.
- **Verification — local.** `VERIFY_MODEL` (`gemma3:12b`) compares the original
  and generated images and decides pass/fail. Using a *separate* model from
  understanding is optional; for a more trustworthy verdict you can point it at a
  stronger vision model (e.g. `kimi-k2.6:cloud`).

### Verification & retries

Generation is image-to-image with a fixed `IMG2IMG_STRENGTH` (default `0.9`),
which is a compromise: too low and the original plate refuses to disappear (the
init image pins it in place — even FLUX at 0.8 keeps the plate); too high and the
food drifts from the original. To catch the former, the pipeline sends **both the
original photo and the generated result** to `VERIFY_MODEL`, which judges by
*comparing* them whether the food has moved off the original plate onto a
different, unconventional surface while still looking like the same food.
Comparing to the original is deliberate: a round food (e.g. a whole pizza) on
any flat unconventional surface can otherwise read as "still on a plate".

- **On failure**, the image is regenerated with a **fresh seed** (the configured
  seed plus the attempt number) and checked again. The diffusion pipeline is
  loaded **once** and reused across attempts, so retries only cost an extra
  render, not another model load.
- **Attempts** = 1 render + `VERIFY_RETRIES` retries (so the default `3` means
  up to 4 renders).
- Configure per-invocation with `--retries N`, `--verify-model MODEL`, or
  `--strength X`; disable the check entirely with `--no-verify`.
- Each attempt costs one extra vision call to `VERIFY_MODEL` (free when it's a
  local model).

> **Note on verifiers.** The verifier must be vision-capable through Ollama.
> `gemma4:*` MLX builds are text-only (no `vision`), so they can't be used.
> `glm-5.2:cloud` is also not vision-capable via Ollama. Use `gemma3:12b`
> (default, local) or a cloud vision model like `kimi-k2.6:cloud` instead.

---

## Development

```bash
# Run the test suite (no MLX/PyTorch required)
pytest
```

Layout:

```
wewantnoplates/
├── wewantnoplates/
│   ├── config.py       # all editable knobs (resolution lives here)
│   ├── cli.py          # Click CLI
│   ├── pipeline.py     # orchestrates the stages
│   ├── understand.py   # cloud vision -> transformation prompt (Ollama)
│   ├── generate.py     # local image-to-image generator (pluggable backend)
│   └── io_util.py      # load file/URL, size math, save
├── tests/              # unit tests
└── pyproject.toml
```

### Swapping the generator backend

`generate.py` exposes an `ImageGenerator` interface:

```python
def generate(self, image: Image, prompt: str, width: int, height: int, seed) -> Image
```

Add a new backend (e.g. an MLX image-to-image model) by implementing that
signature and pointing `GENERATOR_BACKEND` at it (default: `diffusers`).

---

## Known limitations

- The generation step needs the `diffusers`/`torch` stack; `--dry-run` and the
  understanding path work with just the base install.
- Cloud vision models (if you opt into them via `--vision-model`/`--verify-model`)
  may require Ollama **usage/balance** and can be billed; a `402` means the model
  needs usage. The default `gemma3:12b` is fully local and free.
- `IMG2IMG_STRENGTH` balances fidelity vs. transformation: too high and the food
  drifts from the original; too low and the plate may not fully disappear.
  FLUX.1-schnell at the default `0.9` removes the plate; at `0.8` it tends to keep
  the original plate (the init image pins it in place). The verify/retry loop
  catches residual failures automatically.
- Verification compares the original and generated images and is told that a
  round food on an unconventional surface is not a plate. If a verifier is still
  stubborn for a round dish, switch `--verify-model` or eyeball the saved PNG.
