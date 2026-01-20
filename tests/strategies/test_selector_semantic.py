# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import patch

import numpy as np

from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import SemanticSelector


class MockEmbeddingProvider:
    """Mock provider returning deterministic embeddings."""

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        results = []
        for text in texts:
            # Expect format "q: <number>"
            try:
                val = float(text.split(": ")[1])
                # Return 2D point
                results.append([val, val])
            except (IndexError, ValueError):
                results.append([0.0, 0.0])
        return results


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
        # Assign all points to cluster 0 or 1
        instance.labels_ = np.array([0] * 5 + [1] * 5)
        # Centers for 5 clusters (indices 0..4)
        instance.cluster_centers_ = np.zeros((5, 2))

        # When select runs:
        # i=0: labels==0 has points -> select one
        # i=1: labels==1 has points -> select one
        # i=2,3,4: empty -> continue
        # Total selected from clustering = 2
        # Need 5. Should fill 3 more randomly.

        selected = selector.select(ds, k=5)

        assert len(selected) == 5
        # Verify uniqueness?
        # Inputs should be unique if logic works
        inputs = [ex.inputs["q"] for ex in selected]
        assert len(set(inputs)) == 5
