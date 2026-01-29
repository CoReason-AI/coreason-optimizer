# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

"""
Dataset loading and manipulation utilities.

This module provides the Dataset class to load training data from CSV or JSONL files
and split it into training, validation, and test sets.
"""

import csv
import json
import random
from collections.abc import Iterator
from pathlib import Path

from coreason_identity.models import UserContext

from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.utils.logger import logger


class Dataset:
    """A container for training data with loading and splitting capabilities."""

    def __init__(self, examples: list[TrainingExample]):
        """
        Initialize the Dataset.

        Args:
            examples: A list of TrainingExample objects.
        """
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.examples)

    @classmethod
    def from_csv(
        cls,
        filepath: str | Path,
        input_cols: list[str],
        reference_col: str,
        *,
        context: UserContext,
    ) -> "Dataset":
        """
        Load a dataset from a CSV file.

        Args:
            filepath: Path to the CSV file.
            input_cols: List of column names to treat as inputs.
            reference_col: Column name to treat as the reference output.
            context: The user context authorizing this operation.

        Returns:
            A Dataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If context is missing.
        """
        if context is None:
            raise ValueError("UserContext is required.")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(
            "Loading dataset from CSV",
            user_id=context.user_id,
            filepath=str(path),
        )

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
    def from_jsonl(cls, filepath: str | Path, *, context: UserContext) -> "Dataset":
        """
        Load a dataset from a JSONL file.

        Expected format per line:
        {"inputs": {...}, "reference": ...}
        or
        {"input": ..., "output": ...} (will be normalized)

        Args:
            filepath: Path to the JSONL file.
            context: The user context authorizing this operation.

        Returns:
            A Dataset instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If context is missing.
        """
        if context is None:
            raise ValueError("UserContext is required.")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(
            "Loading dataset from JSONL",
            user_id=context.user_id,
            filepath=str(path),
        )

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
        """
        Split the dataset into Train, Validation, and Test sets.

        Args:
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.
            seed: Random seed for shuffling.

        Returns:
            A tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            ValueError: If train_ratio + val_ratio > 1.0.
        """
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
