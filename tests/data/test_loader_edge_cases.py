# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import csv
import json
import tempfile
from pathlib import Path

import pytest
from coreason_identity.models import UserContext

from coreason_optimizer.data.loader import Dataset


def test_csv_missing_columns(mock_context: UserContext) -> None:
    """Test that missing columns in CSV results in skipped rows or empty inputs."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["q", "a"])
        # Row with missing value for 'a' (CSV reader might interpret this based on structure)
        # If we ask for 'q' and 'a', but row has only 1 val, DictReader usually handles it mapping rest to None
        writer.writerow(["only_q"])
        filepath = Path(f.name)

    try:
        # csv.DictReader behavior: if row has fewer fields than fieldnames, values are None
        ds = Dataset.from_csv(filepath, input_cols=["q"], reference_col="a", context=mock_context)
        # Should skip because reference 'a' is None
        assert len(ds) == 0
    finally:
        filepath.unlink()


def test_csv_empty_file(mock_context: UserContext) -> None:
    """Test loading an empty CSV file (header only or completely empty)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("q,a\n")  # Header only
        filepath = Path(f.name)

    try:
        ds = Dataset.from_csv(filepath, input_cols=["q"], reference_col="a", context=mock_context)
        assert len(ds) == 0
    finally:
        filepath.unlink()


def test_jsonl_malformed_line(mock_context: UserContext) -> None:
    """Test that malformed JSON lines cause a failure (or check specific behavior)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"q": "good", "reference": "ok"}\n')
        f.write("INVALID JSON\n")
        filepath = Path(f.name)

    try:
        # Current implementation uses json.loads inside a loop without try-except block for parsing
        # So it should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            Dataset.from_jsonl(filepath, context=mock_context)
    finally:
        filepath.unlink()


def test_split_tiny_dataset() -> None:
    """Test splitting a dataset with 0 or 1 items."""
    # Empty
    ds_empty = Dataset([])
    t, v, te = ds_empty.split()
    assert len(t) == 0
    assert len(v) == 0
    assert len(te) == 0

    # Single item
    # 0.8 train -> 0.8 * 1 = 0.8 -> int is 0
    # 0.1 val -> 0.1 * 1 = 0.1 -> int(0.8+0.1) = 0
    # test -> rest -> 1
    # This behavior depends on int() truncation.
    # 80% of 1 is 0.

    # If we want to ensure at least 1 in train if possible, we might need ceil or logic,
    # but standard logic is usually strictly ratio based.

    from coreason_optimizer.core.models import TrainingExample

    ds_one = Dataset([TrainingExample(inputs={"a": 1}, reference=1)])
    t, v, te = ds_one.split(train_ratio=0.8, val_ratio=0.1)

    # Check strict math behavior:
    # train_end = int(1 * 0.8) = 0
    # val_end = int(1 * 0.9) = 0
    # test = [0:] -> 1 item

    assert len(t) == 0
    assert len(v) == 0
    assert len(te) == 1

    # Force into train
    t, v, te = ds_one.split(train_ratio=1.0, val_ratio=0.0)
    assert len(t) == 1
