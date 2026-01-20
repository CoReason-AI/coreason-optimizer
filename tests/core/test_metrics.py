# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import pytest
from coreason_optimizer.core.metrics import ExactMatch, F1Score, MetricFactory, normalize_answer


def test_normalize_answer() -> None:
    assert normalize_answer("The quick Brown Fox!") == "quick brown fox"
    assert normalize_answer("  spaces  ") == "spaces"
    assert normalize_answer("a an the") == ""
    # Unicode and Special Chars
    assert normalize_answer("café") == "café"
    assert normalize_answer("ñandú") == "ñandú"
    assert normalize_answer("Hello\u00a0World") == "hello world"  # Non-breaking space
    assert normalize_answer("😊") == "😊"  # Emojis remain (not punctuation)


def test_exact_match() -> None:
    em = ExactMatch()
    assert em("Hello World", "hello world") == 1.0
    assert em("Hello World", "Hello World!") == 1.0  # Punctuation ignored
    assert em("foo", "bar") == 0.0
    assert em("123", 123) == 1.0


def test_exact_match_multiple_references() -> None:
    em = ExactMatch()
    # Matches one of the references
    assert em("foo", ["bar", "foo", "baz"]) == 1.0
    # Matches none
    assert em("qux", ["bar", "foo"]) == 0.0
    # Empty list
    assert em("foo", []) == 0.0


def test_f1_score() -> None:
    f1 = F1Score()
    # Perfect match
    assert f1("hello world", "hello world") == 1.0
    # No match
    assert f1("foo", "bar") == 0.0
    # Partial match
    # pred: "cat sat" (2 tokens), ref: "cat sat mat" (3 tokens)
    # common: 2. precision: 2/2=1.0. recall: 2/3=0.66. f1: 2*1*0.66 / 1.66 = 1.33 / 1.66 = 0.8
    assert f1("cat sat", "cat sat mat") == pytest.approx(0.8)

    # Empty cases
    assert f1("", "") == 1.0
    assert f1("foo", "") == 0.0
    assert f1("", "foo") == 0.0


def test_f1_score_multiple_references() -> None:
    f1 = F1Score()
    # Should take the max score
    # "cat" vs "cat" -> 1.0
    # "cat" vs "dog" -> 0.0
    assert f1("cat", ["dog", "cat"]) == 1.0

    # Partial matches
    # "cat sat" vs "cat sat mat" -> 0.8
    # "cat sat" vs "cat sat" -> 1.0
    # Should pick 1.0
    assert f1("cat sat", ["cat sat mat", "cat sat"]) == 1.0

    # Empty list
    assert f1("cat", []) == 0.0


def test_metric_factory() -> None:
    assert isinstance(MetricFactory.get("exact_match"), ExactMatch)
    assert isinstance(MetricFactory.get("f1_score"), F1Score)

    with pytest.raises(ValueError):
        MetricFactory.get("unknown")


def test_non_string_types() -> None:
    em = ExactMatch()
    f1 = F1Score()

    # Integers
    assert em("123", 123) == 1.0
    assert f1("123", 123) == 1.0

    # Floats
    assert em("123.45", 123.45) == 1.0

    # None (prediction is typed as str, but if it comes in as None?)
    # normalize_answer expects str. In Python runtime, this would raise AttributeError.
    # We follow the type hint `prediction: str`.
    # However, reference can be Any.

    # Reference as None -> str(None) -> "None"
    assert em("None", None) == 1.0
    assert em("foo", None) == 0.0
