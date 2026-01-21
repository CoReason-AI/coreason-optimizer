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
from pathlib import Path

import pytest

from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.data.loader import Dataset


def test_dataset_mixed_jsonl_formats(tmp_path: Path) -> None:
    """Test loading a JSONL file with mixed schema conventions."""
    p = tmp_path / "mixed.jsonl"
    lines = [
        # Standard Format 1
        {"inputs": {"q": "1"}, "reference": "a"},
        # Standard Format 2
        {"input": {"q": "2"}, "output": "b"},
        # Fallback Format (keys treated as inputs, explicit reference ignored/missing here?)
        # Wait, loader logic: reference = data.pop("reference", data.pop("output", None))
        # If no ref/output, it skips.
        # Let's try one that works with fallback but has no explicit input key (uses remaining)
        {"q": "3", "reference": "c"},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    ds = Dataset.from_jsonl(p)
    assert len(ds) == 3

    # Check item 1
    assert ds[0].inputs == {"q": "1"}
    assert ds[0].reference == "a"

    # Check item 2
    assert ds[1].inputs == {"q": "2"}
    assert ds[1].reference == "b"

    # Check item 3
    assert ds[2].inputs == {"q": "3"}
    assert ds[2].reference == "c"


def test_dataset_skipping_invalid_rows(tmp_path: Path) -> None:
    """Test that rows without reference/output are skipped in JSONL."""
    p = tmp_path / "invalid.jsonl"
    lines = [
        {"q": "valid", "reference": "a"},
        {"q": "missing_ref"},  # Should be skipped
        {"input": "valid2", "output": "b"},
        {"inputs": {"q": "missing_ref_2"}},  # Should be skipped
    ]
    with open(p, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    ds = Dataset.from_jsonl(p)
    assert len(ds) == 2
    assert ds[0].reference == "a"
    assert ds[1].reference == "b"


def test_budget_manager_edge_cases() -> None:
    """Test edge cases for BudgetManager."""
    # Negative budget
    with pytest.raises(ValueError, match="Budget limit must be positive"):
        BudgetManager(-1.0)

    # Zero budget (should fail based on implementation `gt=0` in pydantic or check in init)
    # The class init says: if budget_limit_usd <= 0: raise ValueError
    with pytest.raises(ValueError, match="Budget limit must be positive"):
        BudgetManager(0.0)

    # Valid budget
    bm = BudgetManager(10.0)
    assert bm.total_cost_usd == 0.0

    # Consume negative usage (should raise ValueError)
    from coreason_optimizer.core.interfaces import UsageStats

    with pytest.raises(ValueError, match="Usage stats cannot be negative"):
        bm.consume(UsageStats(cost_usd=-0.1))
