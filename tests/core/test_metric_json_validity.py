# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from coreason_optimizer.core.metrics import JsonValidity, MetricFactory


def test_json_validity_simple() -> None:
    jv = JsonValidity()
    # Reference is ignored
    assert jv('{"a": 1}', "irrelevant") == 1.0
    assert jv('["a", "b"]', None) == 1.0
    assert jv("123", []) == 1.0  # Valid JSON number
    assert jv('"string"', {}) == 1.0  # Valid JSON string
    assert jv("true", None) == 1.0
    assert jv("null", None) == 1.0


def test_json_validity_invalid() -> None:
    jv = JsonValidity()
    assert jv("{a: 1}", None) == 0.0  # Missing quotes
    assert jv('{"a": 1', None) == 0.0  # Missing brace
    assert jv("random text", None) == 0.0
    assert jv("", None) == 0.0


def test_json_validity_markdown() -> None:
    jv = JsonValidity()

    # Standard json block
    s1 = """
    Here is the output:
    ```json
    {"key": "value"}
    ```
    Hope it helps.
    """
    assert jv(s1, None) == 1.0

    # Block without lang
    s2 = """
    ```
    [1, 2, 3]
    ```
    """
    assert jv(s2, None) == 1.0

    # Multiple blocks (should take first)
    s3 = """
    ```json
    {"a": 1}
    ```
    ```json
    {"b": 2}
    ```
    """
    assert jv(s3, None) == 1.0

    # Malformed block (no closing fence)
    # The regex checks for opening and closing. If closing missing, it fails to match regex,
    # falls back to full text, which fails parsing.
    s4 = """
    ```json
    {"a": 1}
    """
    assert jv(s4, None) == 0.0


def test_json_validity_factory() -> None:
    assert isinstance(MetricFactory.get("json_validity"), JsonValidity)


def test_json_validity_whitespace() -> None:
    jv = JsonValidity()
    assert jv('  {"a": 1}  ', None) == 1.0
    assert jv(' \n {"a": 1} \t ', None) == 1.0


def test_json_validity_dotall() -> None:
    jv = JsonValidity()
    # Ensure regex matches across lines
    s = """
    ```json
    {
        "a": 1
    }
    ```
    """
    assert jv(s, None) == 1.0
