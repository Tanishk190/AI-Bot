"""Tests for the LLM JSON parsing helper — the fragile core every
LLM-as-JSON tool (PII/Financial Extractor, classifier) depends on."""
import pytest

from core.llm import parse_llm_json


def test_plain_json_object():
    assert parse_llm_json('{"name": "Acme", "n": 3}') == {"name": "Acme", "n": 3}


def test_json_fenced_block():
    raw = '```json\n{"a": 1, "b": 2}\n```'
    assert parse_llm_json(raw) == {"a": 1, "b": 2}


def test_surrounding_whitespace():
    assert parse_llm_json('   {"ok": true}   ') == {"ok": True}


def test_nested_structure():
    raw = '{"identifiers": ["PAN: ABCDE1234F"], "amounts": []}'
    assert parse_llm_json(raw) == {"identifiers": ["PAN: ABCDE1234F"], "amounts": []}


def test_invalid_json_raises_valueerror():
    with pytest.raises(ValueError):
        parse_llm_json("this is not json at all")


def test_bare_fence_block():
    # A bare ``` fence (no 'json' tag) is now stripped correctly.
    assert parse_llm_json('```\n{"a": 1}\n```') == {"a": 1}
