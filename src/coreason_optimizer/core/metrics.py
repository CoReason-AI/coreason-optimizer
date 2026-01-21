# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import collections
import json
import re
import string
from typing import Any, Callable

from coreason_optimizer.core.interfaces import Metric


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class ExactMatch(Metric):
    """Computes whether the prediction exactly matches the reference (after normalization)."""

    def _score_single(self, prediction: str, reference: Any) -> float:
        return 1.0 if normalize_answer(prediction) == normalize_answer(str(reference)) else 0.0

    def __call__(self, prediction: str, reference: Any, **kwargs: Any) -> float:
        if isinstance(reference, list):
            return max((self._score_single(prediction, ref) for ref in reference), default=0.0)
        return self._score_single(prediction, reference)


class F1Score(Metric):
    """Computes F1 score based on token overlap."""

    def _score_single(self, prediction: str, reference: Any) -> float:
        prediction_tokens = normalize_answer(prediction).split()
        reference_tokens = normalize_answer(str(reference)).split()

        common = collections.Counter(prediction_tokens) & collections.Counter(reference_tokens)
        num_same = sum(common.values())

        if len(prediction_tokens) == 0 or len(reference_tokens) == 0:
            return int(prediction_tokens == reference_tokens)

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    def __call__(self, prediction: str, reference: Any, **kwargs: Any) -> float:
        if isinstance(reference, list):
            return max((self._score_single(prediction, ref) for ref in reference), default=0.0)
        return self._score_single(prediction, reference)


class JsonValidity(Metric):
    """Computes whether the prediction is valid JSON (ignoring reference)."""

    def __call__(self, prediction: str, reference: Any, **kwargs: Any) -> float:
        # 1. Clean up whitespace
        text = prediction.strip()

        # 2. Extract content from Markdown code blocks if present
        # Matches ```json ... ``` or ``` ... ``` (multiline)
        # We search for the *first* block.
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)

        # 3. Try to parse
        try:
            json.loads(text)
            return 1.0
        except json.JSONDecodeError:
            return 0.0


class MetricFactory:
    """Factory for creating metrics by name."""

    _metrics: dict[str, Callable[[], Metric]] = {
        "exact_match": ExactMatch,
        "f1_score": F1Score,
        "json_validity": JsonValidity,
    }

    @classmethod
    def get(cls, name: str) -> Metric:
        """Get a metric instance by name."""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. Available: {list(cls._metrics.keys())}")
        return cls._metrics[name]()
