# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from coreason_optimizer.core.metrics import JsonValidity


def test_json_validity_simple() -> None:
    metric = JsonValidity()
    assert metric('{"a": 1}', None) == 1.0
    assert metric("invalid", None) == 0.0


def test_json_validity_markdown_explicit() -> None:
    metric = JsonValidity()
    # Explicit json block
    text = 'Here is the json:\n```json\n{"a": 1}\n```'
    assert metric(text, None) == 1.0

    # Invalid inside block
    text_bad = "```json\n{a: 1}\n```"
    assert metric(text_bad, None) == 0.0


def test_json_validity_markdown_generic() -> None:
    metric = JsonValidity()
    # Generic block
    text = '```\n{"a": 1}\n```'
    assert metric(text, None) == 1.0

    # Python block (should work if content is valid json)
    text_py = '```python\n{"a": 1}\n```'
    assert metric(text_py, None) == 1.0


def test_json_validity_multiple_blocks() -> None:
    metric = JsonValidity()
    # First valid wins
    text = '```json\n{"a": 1}\n```\nAnd another:\n```json\ninvalid\n```'
    assert metric(text, None) == 1.0

    # If first is invalid, keep looking
    text_2 = '```json\ninvalid\n```\n```json\n{"b": 2}\n```'
    assert metric(text_2, None) == 1.0
