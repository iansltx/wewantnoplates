"""Cloud "understanding" step.

Sends the input image to a vision-capable model running through Ollama (in this
project's setup that's a cloud-hosted model like `minimax-m3:cloud`) and asks it
to write a **transformation prompt** — a short instruction that will be fed to
the image-to-image generator alongside the *original* image.

The prompt only describes the *change* to apply (move the food off the plate
onto an absurd surface, in r/wewantplates style). It deliberately does NOT
describe the food from scratch; the generator already sees the image.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import config

# Instructions to the vision model. It must return strict JSON.
SYSTEM_PROMPT = """\
You are a creative food stylist producing content for the r/wewantplates \
subreddit, which celebrates food being served on absurd, unconventional \
surfaces instead of plates (wooden boards, bricks, power tools, shoes, sinks, \
leaf blowers, etc.).

You will be given a photo of food on a plate. Write ONE image-to-image \
TRANSFORMATION prompt to be applied to the photo. The generator already sees \
the photo, so:
  - Do NOT describe the food or scene from scratch (no re-listing ingredients, \
colors, or lighting that already exist in the photo).
  - Only state the transformation: move the food OFF the plate and re-serve it \
on a specific, funny, absurd surface in r/wewantplates style. Pick one vivid \
surface and setting (e.g. a rusty garden shovel on a gravel patio), keep the \
food itself faithful to the photo, preserve good lighting and photorealism, and \
add a short punchy caption-style phrase.
  - Keep it under ~60 words.

Return JSON only, with exactly one key:
{"transformation_prompt": "..."}\
"""

_USER_PROMPT = "Write the transformation prompt for this photo."


@dataclass
class Understanding:
    """Result of the cloud understanding step: the transformation prompt."""

    transformation_prompt: str
    raw_response: str


def understand(image_bytes: bytes, vision_model: str | None = None) -> Understanding:
    """Produce an image-to-image transformation prompt via Ollama vision."""
    model = vision_model or config.VISION_MODEL

    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - trivial
        raise SystemExit(
            "The `ollama` python package is required. Install with:\n"
            "    pip install -e .[gen]   (or: pip install ollama)"
        ) from exc

    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT, "images": [image_bytes]},
        ],
        format="json",
        options={"temperature": 0.7},
    )

    content = response["message"]["content"]
    try:
        payload = _extract_json(content)
    except json.JSONDecodeError as exc:  # pragma: no cover - model hiccup
        raise RuntimeError(
            f"Vision model did not return valid JSON:\n{content}"
        ) from exc

    return Understanding(
        transformation_prompt=payload.get("transformation_prompt", "").strip(),
        raw_response=content,
    )


def _extract_json(text: str) -> dict:
    """Parse a JSON object, tolerating markdown code fences and surrounding text."""
    text = text.strip()
    # Strip ```json ... ``` fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first balanced {...} block in the response.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
