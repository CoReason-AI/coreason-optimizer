# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import random
from abc import ABC, abstractmethod

from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset


class BaseSelector(ABC):
    """Abstract base class for few-shot example selection strategies."""

    @abstractmethod
    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """Select k examples from the training set."""
        pass  # pragma: no cover


class RandomSelector(BaseSelector):
    """Randomly selects examples from the training set."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """Select k random examples."""
        if len(trainset) <= k:
            return list(trainset)

        rng = random.Random(self.seed)
        return rng.sample(list(trainset), k)
