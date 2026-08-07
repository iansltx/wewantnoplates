"""Cloud "understanding" + verification steps.

``understand`` sends the input image to a vision-capable model running through
Ollama (in this project's setup that's a cloud-hosted model like
``minimax-m3:cloud``) and asks it to write a **transformation prompt** — a short
instruction fed to the image-to-image generator alongside the *original* image.

The prompt only describes the *change* to apply (move the food off the plate
onto an absurd surface, in r/wewantplates style). It deliberately does NOT
describe the food from scratch; the generator already sees the image.

``verify`` then sends BOTH the original photo and the generated result back to a
(possibly different) vision model and asks for a pass/fail verdict by
*comparing* them — used to retry generation until the plate is actually gone.
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
  - Only state the transformation: take the food OFF the plate and re-serve it \
directly on a specific, funny, absurd surface in r/wewantplates style. Pick \
one vivid surface and setting (e.g. a rusty garden shovel on a gravel patio).
  - Make the plate and any cutlery/tableware DISAPPEAR: the food must rest \
directly on the new surface with no plate underneath. Do NOT tell the generator \
to keep or preserve the original plate, table, or tablecloth — those must be \
gone. The food itself should stay recognizable.
  - Keep it under ~60 words and end with a short punchy caption-style phrase.

Return JSON only, with exactly one key:
{"transformation_prompt": "..."}\
"""

_USER_PROMPT = "Write the transformation prompt for this photo."


@dataclass
class Understanding:
    """Result of the cloud understanding step: the transformation prompt."""

    transformation_prompt: str
    raw_response: str


@dataclass
class Verification:
    """Result of checking the generated image against the original."""

    passed: bool
    reason: str
    raw_response: str


def _ollama_client():
    """Return a configured Ollama client, with a friendly error if missing."""
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - trivial
        raise SystemExit(
            "The `ollama` python package is required. Install with:\n"
            "    pip install -e .[gen]   (or: pip install ollama)"
        ) from exc

    return ollama.Client(
        host=config.OLLAMA_HOST, timeout=config.UNDERSTAND_TIMEOUT_SECONDS
    )


def understand(image_bytes: bytes, vision_model: str | None = None) -> Understanding:
    """Produce an image-to-image transformation prompt via Ollama vision."""
    model = vision_model or config.VISION_MODEL
    client = _ollama_client()
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


# Instructions for the quality-control step. It compares the original and
# generated images and must return strict JSON with a pass/fail verdict.
_VERIFY_SYSTEM_PROMPT = """\
You are a quality-control evaluator for an image-to-image tool in the \
r/wewantplates style: it re-serves food that was on a plate onto an absurd, \
unconventional surface (wooden boards, bricks, power tools, shoes, sinks, \
toilet seats, etc.).

You will be shown TWO images: the ORIGINAL (food on a plate) and the GENERATED \
result. Compare them and make a single PASS/FAIL decision:
  PASS if BOTH are true:
    - The food in the generated image rests on a DIFFERENT surface than the \
original plate/tableware. The original plate is gone and the food is on \
something unconventional.
    - The food is still recognizable as the same food.
  The specific surface in the prompt (e.g. "rusty garden shovel") is an \
inspiration; exact props do NOT need to match literally. What matters is that \
the surface clearly changed from the original plate to something \
unconventional.
  FAIL only if the generated image still shows the food on the SAME plate or \
tableware as the original, or the food has become unrecognizable.
  IMPORTANT: a round food (e.g. a whole pizza) sitting on any flat \
unconventional surface is NOT "on a plate" merely because it is round. Judge \
by whether the surface differs from the original plate, not by shape alone.

Return JSON only, with exactly these keys:
{"passed": true_or_false, "reason": "one short sentence explaining the verdict"}\
"""


def verify(
    generated_bytes: bytes,
    original_bytes: bytes,
    transformation_prompt: str,
    verify_model: str | None = None,
) -> Verification:
    """Check whether a generated image moved the food off the original plate.

    Sends BOTH the original photo and the generated result to the cloud vision
    model and asks for a pass/fail verdict by *comparing* them. Comparing to the
    original is more reliable than judging the result in isolation, because a
    round food (e.g. a whole pizza) re-served on any flat unconventional surface
    can otherwise read as "still on a plate".
    """
    model = verify_model or config.VERIFY_MODEL
    client = _ollama_client()
    user_prompt = (
        "Image 1 is the ORIGINAL photo (food on a plate).\n"
        "Image 2 is the GENERATED result.\n\n"
        f"The transformation prompt was:\n{transformation_prompt}\n\n"
        "Comparing the two images: has the food been moved OFF the original "
        "plate onto a clearly DIFFERENT, unconventional surface, while still "
        "looking like the same food?"
    )
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
                "images": [original_bytes, generated_bytes],
            },
        ],
        format="json",
        options={"temperature": 0.0},
    )

    content = response["message"]["content"]
    try:
        payload = _extract_json(content)
    except json.JSONDecodeError as exc:  # pragma: no cover - model hiccup
        raise RuntimeError(
            f"Vision model did not return valid JSON:\n{content}"
        ) from exc

    return Verification(
        passed=bool(payload.get("passed")),
        reason=str(payload.get("reason", "")).strip(),
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