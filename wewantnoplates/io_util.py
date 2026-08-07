"""Image input/output helpers: load from a local path or URL, compute the
constrained target size, and save results.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from . import config
from . import __version__

# Some hosts (e.g. Wikimedia) reject requests with no/blank User-Agent.
_HEADERS = {"User-Agent": f"wewantnoplates/{__version__} (image transform tool)"}

# FLUX (and most diffusion backends) require the width/height to be a multiple
# of 8. We round the derived short side up to the nearest multiple of 8.
BLOCK_MULTIPLE = 8


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def load_image(source: str) -> Image.Image:
    """Load a Pillow image from either a local file path or a URL."""
    if _is_url(source):
        resp = requests.get(source, timeout=120, headers=_HEADERS)
        resp.raise_for_status()
        raw = io.BytesIO(resp.content)
    else:
        raw = Path(source).open("rb")
    with raw:
        image = Image.open(raw)
        image.load()
    return image


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a Pillow image to bytes (used for the Ollama vision call)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def compute_target_size(
    src_w: int, src_h: int, max_side: int | None = None
) -> tuple[int, int]:
    """Return (width, height) for the output image.

    The longest side is capped at ``max_side`` pixels (default from
    ``config.MAX_SIDE_PIXELS``); the shorter side is scaled to preserve the
    source aspect ratio, then rounded to a multiple of 8.
    """
    if max_side is None:
        max_side = config.MAX_SIDE_PIXELS
    max_side = max(int(max_side), BLOCK_MULTIPLE)

    if src_w >= src_h:
        w, h = max_side, round(max_side * src_h / src_w)
    else:
        h, w = max_side, round(max_side * src_w / src_h)

    w = max(BLOCK_MULTIPLE, (w // BLOCK_MULTIPLE) * BLOCK_MULTIPLE)
    h = max(BLOCK_MULTIPLE, (h // BLOCK_MULTIPLE) * BLOCK_MULTIPLE)
    return w, h


def make_output_path(extension: str = ".png", out_dir: str | None = None) -> Path:
    """Build a unique output file path under the configured output directory."""
    if out_dir is None:
        out_dir = config.OUTPUT_DIR
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return directory / f"wewantnoplates-{stamp}{extension}"
