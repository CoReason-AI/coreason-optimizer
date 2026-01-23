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

from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset


def test_dataset_initialization() -> None:
    examples = [
        TrainingExample(inputs={"q": "1"}, reference="A"),
        TrainingExample(inputs={"q": "2"}, reference="B"),
    ]
    ds = Dataset(examples)
    assert len(ds) == 2
    assert ds[0].reference == "A"
    assert [e.reference for e in ds] == ["A", "B"]


def test_load_from_csv() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "extra"])
        writer.writerow(["What is 1+1?", "2", "ignore"])
        writer.writerow(["What is 2+2?", "4", "ignore"])
        # Add invalid rows
        writer.writerow(["", "broken", "ignore"])  # Missing question
        writer.writerow(["q", "", "ignore"])  # Missing answer
        filepath = Path(f.name)

    try:
        ds = Dataset.from_csv(filepath, input_cols=["question"], reference_col="answer")
        assert len(ds) == 2
        assert ds[0].inputs["question"] == "What is 1+1?"
        assert ds[0].reference == "2"
        assert ds[0].metadata["source"] == str(filepath)
    finally:
        filepath.unlink()


def test_load_from_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        Dataset.from_csv("non_existent.csv", [], "")


def test_load_from_jsonl() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"inputs": {"q": "foo"}, "reference": "bar"}) + "\n")
        f.write(json.dumps({"input": {"q": "baz"}, "output": "qux"}) + "\n")
        f.write(json.dumps({"q": "simple", "reference": "simple_ref"}) + "\n")
        # Add invalid rows
        f.write(json.dumps({"q": "no_ref"}) + "\n")  # Missing reference
        filepath = Path(f.name)

    try:
        ds = Dataset.from_jsonl(filepath)
        assert len(ds) == 3

        # Check first format
        assert ds[0].inputs["q"] == "foo"
        assert ds[0].reference == "bar"

        # Check second format
        assert ds[1].inputs["q"] == "baz"
        assert ds[1].reference == "qux"

        # Check third format
        assert ds[2].inputs["q"] == "simple"
        assert ds[2].reference == "simple_ref"
    finally:
        filepath.unlink()


def test_load_from_jsonl_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        Dataset.from_jsonl("non_existent.jsonl")


def test_split_dataset() -> None:
    examples = [TrainingExample(inputs={"q": str(i)}, reference=str(i)) for i in range(100)]
    ds = Dataset(examples)

    train, val, test = ds.split(train_ratio=0.8, val_ratio=0.1)

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Ensure no overlap (simple check on references)
    train_refs = {e.reference for e in train}
    val_refs = {e.reference for e in val}
    test_refs = {e.reference for e in test}

    assert train_refs.isdisjoint(val_refs)
    assert train_refs.isdisjoint(test_refs)
    assert val_refs.isdisjoint(test_refs)


def test_split_invalid_ratios() -> None:
    ds = Dataset([])
    with pytest.raises(ValueError):
        ds.split(train_ratio=0.9, val_ratio=0.2)
