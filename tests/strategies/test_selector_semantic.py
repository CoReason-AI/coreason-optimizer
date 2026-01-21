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
from unittest.mock import patch

import numpy as np

from coreason_optimizer.core.interfaces import EmbeddingProvider, EmbeddingResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import SemanticSelector


class MockEmbeddingProvider:
    """Mock provider returning deterministic embeddings."""

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        results = []
        for text in texts:
            # Expect JSON format
            try:
                data = json.loads(text)
                # We assume examples have 'q' input key for these tests
                val_str = str(data.get("q", "0"))
                val = float(val_str)
                # Return 2D point
                results.append([val, val])
            except (ValueError, TypeError, json.JSONDecodeError):
                results.append([0.0, 0.0])

        return EmbeddingResponse(embeddings=results, usage=UsageStats())


def test_semantic_selector_clustering() -> None:
    # Create examples that form distinct clusters
    # Cluster 1: 1, 1.1, 0.9 (Centroid approx 1.0)
    # Cluster 2: 10, 10.1, 9.9 (Centroid approx 10.0)
    vals = [1, 1.1, 0.9, 10, 10.1, 9.9]
    examples = [TrainingExample(inputs={"q": str(v)}, reference=str(v)) for v in vals]
    ds = Dataset(examples)

    provider = MockEmbeddingProvider()
    selector = SemanticSelector(embedding_provider=provider, seed=42)

    # Select k=2. Should pick one from each cluster (approx 1 and 10)
    selected = selector.select(ds, k=2)

    assert len(selected) == 2

    # Check values
    sel_vals = [float(ex.inputs["q"]) for ex in selected]
    # We expect one near 1 and one near 10
    has_low = any(0.8 <= v <= 1.2 for v in sel_vals)
    has_high = any(9.8 <= v <= 10.2 for v in sel_vals)

    assert has_low, f"Expected value near 1, got {sel_vals}"
    assert has_high, f"Expected value near 10, got {sel_vals}"


def test_semantic_selector_small_dataset() -> None:
    examples = [TrainingExample(inputs={"q": "1"}, reference="1")]
    ds = Dataset(examples)
    provider = MockEmbeddingProvider()
    selector = SemanticSelector(provider)

    selected = selector.select(ds, k=5)
    assert len(selected) == 1
    assert selected[0].inputs["q"] == "1"


def test_semantic_selector_fallback() -> None:
    # Request k=3 from 2 examples -> should return 2
    vals = [1, 10]
    examples = [TrainingExample(inputs={"q": str(v)}, reference=str(v)) for v in vals]
    ds = Dataset(examples)
    provider = MockEmbeddingProvider()
    selector = SemanticSelector(provider)

    selected = selector.select(ds, k=3)
    assert len(selected) == 2


def test_semantic_selector_fill_logic() -> None:
    # Trigger fallback logic when we get fewer selected than k
    # We mock KMeans to simulate empty clusters

    examples = [TrainingExample(inputs={"q": str(i)}, reference=str(i)) for i in range(10)]
    ds = Dataset(examples)
    provider = MockEmbeddingProvider()
    selector = SemanticSelector(embedding_provider=provider, seed=42)

    with patch("coreason_optimizer.strategies.selector.KMeans") as MockKMeans:
        instance = MockKMeans.return_value
        instance.fit.return_value = None
        # Mock labels: simulate that only clusters 0 and 1 have points
        # k=5. Clusters 2,3,4 are empty.
        instance.labels_ = np.array([0] * 5 + [1] * 5)
        # Centers for 5 clusters (indices 0..4)
        instance.cluster_centers_ = np.zeros((5, 2))

        selected = selector.select(ds, k=5)

        assert len(selected) == 5
        # Verify uniqueness
        inputs = [ex.inputs["q"] for ex in selected]
        assert len(set(inputs)) == 5


def test_semantic_selector_edge_cases() -> None:
    # Test handling of duplicate examples

    ex1 = TrainingExample(inputs={"q": "A"}, reference="A")
    ex2 = TrainingExample(inputs={"q": "A"}, reference="A")  # Duplicate
    ex3 = TrainingExample(inputs={"q": "A"}, reference="A")  # Duplicate
    ex4 = TrainingExample(inputs={"q": "B"}, reference="B")

    examples = [ex1, ex2, ex3, ex4]
    ds = Dataset(examples)

    # Mock provider
    class EdgeMockProvider(EmbeddingProvider):
        def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
            res = []
            for t in texts:
                # "q": "A" in json
                if '"A"' in t:
                    res.append([0.0, 0.0])
                else:
                    res.append([1.0, 1.0])
            return EmbeddingResponse(embeddings=res, usage=UsageStats())

    selector = SemanticSelector(embedding_provider=EdgeMockProvider(), seed=42)

    # Select k=2. Should pick one A and one B.
    selected = selector.select(ds, k=2)
    assert len(selected) == 2
    inputs = sorted([ex.inputs["q"] for ex in selected])
    assert inputs == ["A", "B"]

    # Select k=3. Should pick A, B, and fill 3rd
    selected = selector.select(ds, k=3)
    assert len(selected) == 3
    inputs = sorted([ex.inputs["q"] for ex in selected])
    assert inputs == ["A", "A", "B"]
