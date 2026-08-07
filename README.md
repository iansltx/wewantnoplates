# wewantnoplates

Turn a photo of food *on a plate* into a photo of the same food defiantly
**NOT on a plate** — in the proud tradition of
[/r/wewantplates](https://www.reddit.com/r/wewantplates/). Give it a pizza on
ceramic; get back the same pizza balanced on a rusty garden shovel, a
circular-saw blade, or a dirty hubcap.

It's an **image-to-image** CLI pipeline. The generator's two inputs are the
**original photo** and a short **transformation prompt**; the output is the
edited image. The cloud vision model's only job is to write that transformation
prompt — it never describes the food from scratch (the generator already sees
the image).

```
  ┌──────────────┐
  │ input image  │──────────────┐
  │ (file / URL) │              │
  └──────────────┘              ▼
         │            ┌──────────────────────────┐
         │            │ cloud vision (Ollama)    │
         │            │ writes transformation    │──▶ sidecar prompt
         │            │ prompt (img2img)         │
         │            └──────────────────────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
        ┌──────────────────────────┐
        │ local image-to-image     │──▶ transformed image saved locally
        │ diffusion (diffusers)    │
        └──────────────────────────┘
```

---

## Requirements

- **Ollama** running (defaults to `http://localhost:11434`) with a
  **vision-capable** model. This project is set up for cloud-hosted models (see
  `ollama list`); the default is `minimax-m3:cloud`, which has `vision` in its
  capabilities.
- **Python 3.10+** for the understanding path. The generation step additionally
  needs the `diffusers`/`torch` stack, which works on **Apple Silicon (MPS)** or
  a **CUDA GPU** (comfortably within a 64 GB machine).
- First generation run downloads the model weights from Hugging Face (a few GB).

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

# Understanding + prompt only (no rendering) — handy for testing
wewantnoplates my_steak_dinner.jpg --dry-run
```

Typical output:

```
Transformation prompt:
  Lift the pizza off the plate and re-serve it on a rusty vintage chrome hubcap
  sitting on a dusty concrete garage floor, crumbs and grease smudged around the
  rim, warm overhead tungsten light, photorealistic, "fresh from the wheel."

Output size: 1024x768
Prompt saved: output/wewantnoplates-<timestamp>.json
Image saved:  output/wewantnoplates-<timestamp>.png
```

---

## How the two inputs combine

The generator is image-to-image:

- **Input image** — the original photo, resized to the target size.
- **Transformation prompt** — a short instruction (≤ ~60 words) that only
  states the *change*: move the food off the plate and re-serve it on a specific
  absurd surface in r/wewantplates style. It does **not** describe the food,
  ingredients, or existing scene.

The cloud vision model (Ollama) looks at the photo and writes that
transformation prompt. That prompt is saved verbatim as a sidecar `.json` next
to the output image, so every edit is reproducible and inspectable.

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
| `VISION_MODEL` | `minimax-m3:cloud` | Cloud model that writes the transformation prompt |
| `IMG2IMG_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | Local image-to-image model |
| `IMG2IMG_STRENGTH` | `0.65` | How strongly the edit is applied (0..1) |
| `GENERATOR_STEPS` / `GENERATOR_GUIDANCE` | `40` / `7.5` | Diffusion sampling parameters |
| `GENERATOR_SEED` | `42` | Reproducibility (set `None` for random) |
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

## The two model roles

The pipeline deliberately mixes a **cloud** model for understanding and a
**local** model for generation:

- **Understanding — cloud.** The vision model runs through Ollama but is hosted
  remotely (the Ollama account's cloud models). It writes the transformation
  prompt, at the cost of a (possibly billed) API. Vision-capable models already
  installed here include `minimax-m3:cloud`, `kimi-k3:cloud`, and
  `kimi-k2.7-code:cloud`.
- **Generation — local.** The image-to-image edit happens entirely on-device via
  `diffusers`/PyTorch (MPS on Apple Silicon, or CUDA). Nothing leaves the
  machine, which keeps rendering free and private.

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
signature and pointing `GENERATOR_BACKEND` at it. The default is `diffusers`.

---

## Known limitations

- The generation step needs the `diffusers`/`torch` stack; `--dry-run` and the
  understanding path work with just the base install.
- Cloud vision models may require Ollama **usage/balance** (some are billed as
  "extra usage"). If a call fails with a `402`, the model needs usage or a
  different free-tier vision model.
- `IMG2IMG_STRENGTH` balances fidelity vs. transformation: too high and the food
  drifts from the original; too low and the plate may not fully disappear.
