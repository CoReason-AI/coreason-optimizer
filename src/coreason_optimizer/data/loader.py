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
import random
from pathlib import Path
from typing import Any

from coreason_optimizer.core.models import TrainingExample


class Dataset:
    """A container for training data with loading and splitting capabilities."""

    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]

    def __iter__(self) -> Any:
        return iter(self.examples)

    @classmethod
    def from_csv(cls, filepath: str | Path, input_cols: list[str], reference_col: str) -> "Dataset":
        """Load a dataset from a CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        examples = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                inputs = {col: row.get(col) for col in input_cols}
                # Check if inputs are missing
                if any(v is None or v == "" for v in inputs.values()):
                    continue

                reference = row.get(reference_col)
                if reference is None or reference == "":
                    continue

                examples.append(
                    TrainingExample(
                        inputs=inputs,
                        reference=reference,
                        metadata={"source": str(path)},
                    )
                )
        return cls(examples)

    @classmethod
    def from_jsonl(cls, filepath: str | Path) -> "Dataset":
        """Load a dataset from a JSONL file.

        Expected format per line:
        {"inputs": {...}, "reference": ...}
        or
        {"input": ..., "output": ...} (will be normalized)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        examples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # Normalize typical formats
                if "inputs" in data and "reference" in data:
                    inputs = data["inputs"]
                    reference = data["reference"]
                elif "input" in data and "output" in data:
                    inputs = data["input"] if isinstance(data["input"], dict) else {"input": data["input"]}
                    reference = data["output"]
                else:
                    # Generic fallback: treat all keys except 'reference'/'output' as inputs
                    reference = data.pop("reference", data.pop("output", None))
                    if reference is None:
                        # Skipping ambiguous lines
                        continue
                    inputs = data

                examples.append(
                    TrainingExample(
                        inputs=inputs,
                        reference=reference,
                        metadata={"source": str(path)},
                    )
                )
        return cls(examples)

    def split(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
    ) -> tuple["Dataset", "Dataset", "Dataset"]:
        """Split the dataset into Train, Validation, and Test sets."""
        if train_ratio + val_ratio > 1.0:
            raise ValueError("Sum of train and val ratios must be <= 1.0")

        random.seed(seed)
        shuffled = list(self.examples)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = shuffled[:train_end]
        val_data = shuffled[train_end:val_end]
        test_data = shuffled[val_end:]

        return Dataset(train_data), Dataset(val_data), Dataset(test_data)
