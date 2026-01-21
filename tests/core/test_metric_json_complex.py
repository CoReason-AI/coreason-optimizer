# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import json
from typing import Any

from coreason_optimizer.core.metrics import JsonValidity


def test_mixed_blocks_prioritize_json() -> None:
    """Test that the metric finds the JSON block even if other blocks exist."""
    jv = JsonValidity()

    # Python block first, then JSON block
    text = """
    Here is the logic:
    ```python
    def foo():
        return {"a": 1}
    ```
    And here is the output:
    ```json
    {"result": "success"}
    ```
    """
    # Current implementation (finding first block) might fail this if it matches python block.
    # Desired: 1.0
    assert jv(text, None) == 1.0


def test_mixed_blocks_generic_last() -> None:
    """Test that if no JSON block is explicitly marked, it checks others?
    Or strictly checks logic."""
    jv = JsonValidity()

    # Text block first, then Generic block with JSON
    text = """
    Ignore this:
    ```
    Some text
    ```
    Output:
    ```
    {"a": 1}
    ```
    """
    # If we iterate, we might find the second one.
    assert jv(text, None) == 1.0


def test_language_tag_handling() -> None:
    """Test that language tags like ```python are not parsed as content."""
    jv = JsonValidity()

    # If regex captures "python" as content, json.loads fails.
    # If regex handles it, content is clean.
    # But content is python code, so it should fail json.loads anyway.
    # This test confirms that we don't crash and correctly identify invalidity.
    text = """
    ```python
    print("hello")
    ```
    """
    assert jv(text, None) == 0.0


def test_unicode_and_escaping() -> None:
    jv = JsonValidity()
    data = {"emoji": "😊", "unicode": "你好", "escaped": "Line\nBreak", "quote": '"'}
    json_str = json.dumps(data)
    text = f"```json\n{json_str}\n```"
    assert jv(text, None) == 1.0


def test_large_nested_json() -> None:
    jv = JsonValidity()
    # Create deep nesting
    data: Any = {"a": 1}
    for _ in range(100):
        data = {"next": data}

    json_str = json.dumps(data)
    text = f"```json\n{json_str}\n```"
    assert jv(text, None) == 1.0


def test_json_with_comments_fail() -> None:
    """Ensure we are strict (no comments)."""
    jv = JsonValidity()
    text = """
    ```json
    {
        "a": 1 // This is a comment
    }
    ```
    """
    assert jv(text, None) == 0.0


def test_trailing_comma_fail() -> None:
    jv = JsonValidity()
    text = """
    ```json
    {
        "a": 1,
    }
    ```
    """
    assert jv(text, None) == 0.0
