"""Tests for the vision-model JSON extraction helper."""

import pytest

from wewantnoplates.understand import _extract_json


def test_extract_fenced_json():
    text = '```json\n{"description": "a", "transformation_prompt": "b"}\n```'
    assert _extract_json(text) == {"description": "a", "transformation_prompt": "b"}


def test_extract_plain_json():
    text = '{"description": "a", "transformation_prompt": "b"}'
    assert _extract_json(text) == {"description": "a", "transformation_prompt": "b"}


def test_extract_json_embedded_in_text():
    text = "Here you go:\n```json\n{\"description\": \"x\", \"transformation_prompt\": \"y\"}\n```\n\nEnjoy!"
    assert _extract_json(text) == {"description": "x", "transformation_prompt": "y"}


def test_extract_invalid_json_raises():
    with pytest.raises(Exception):
        _extract_json("not json at all")
