# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import MagicMock, patch

import numpy as np

from coreason_optimizer.core.interfaces import EmbeddingProvider, EmbeddingResponse
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import RandomSelector, SemanticSelector


def test_random_selector_k_greater_than_len() -> None:
    """Test behavior when k > dataset length."""
    examples = [TrainingExample(inputs={"q": i}, reference=i) for i in range(3)]
    ds = Dataset(examples)
    selector = RandomSelector()
    selected = selector.select(ds, k=5)
    assert len(selected) == 3
    assert selected == examples


def test_random_selector_subset() -> None:
    """Test subset selection."""
    examples = [TrainingExample(inputs={"q": i}, reference=i) for i in range(10)]
    ds = Dataset(examples)
    selector = RandomSelector(seed=42)
    selected = selector.select(ds, k=4)
    assert len(selected) == 4
    # Check reproducibility
    selector_2 = RandomSelector(seed=42)
    selected_2 = selector_2.select(ds, k=4)
    assert selected == selected_2


def test_semantic_selector_init() -> None:
    """Test initialization."""
    mock_provider = MagicMock(spec=EmbeddingProvider)
    selector = SemanticSelector(embedding_provider=mock_provider, embedding_model="test-model")
    assert selector.embedding_provider == mock_provider
    assert selector.embedding_model == "test-model"


def test_semantic_selector_select_small_dataset() -> None:
    """Test selecting when dataset is smaller than k."""
    examples = [TrainingExample(inputs={"q": i}, reference=i) for i in range(2)]
    ds = Dataset(examples)
    mock_provider = MagicMock(spec=EmbeddingProvider)
    selector = SemanticSelector(embedding_provider=mock_provider)

    selected = selector.select(ds, k=5)
    assert len(selected) == 2
    mock_provider.embed.assert_not_called()  # Should short-circuit


def test_semantic_selector_clustering() -> None:
    """Test clustering logic."""
    examples = [TrainingExample(inputs={"q": i}, reference=i) for i in range(10)]
    ds = Dataset(examples)

    # Mock embeddings: 10 vectors of dim 2
    embeddings = []
    for i in range(5):
        embeddings.append([0.0 + i * 0.1, 0.0])
    for i in range(5):
        embeddings.append([10.0 + i * 0.1, 10.0])

    mock_response = EmbeddingResponse(embeddings=embeddings, total_tokens=100, cost_usd=0.001, usage={})

    mock_provider = MagicMock(spec=EmbeddingProvider)
    mock_provider.embed.return_value = mock_response

    with patch("coreason_optimizer.strategies.selector.KMeans") as MockKMeans:
        mock_kmeans_instance = MagicMock()
        MockKMeans.return_value = mock_kmeans_instance

        # Manually set labels and centers to simulate clustering result
        # 5 points in cluster 0, 5 points in cluster 1
        mock_kmeans_instance.labels_ = np.array([0] * 5 + [1] * 5)
        mock_kmeans_instance.cluster_centers_ = np.array([[0.0, 0.0], [10.0, 10.0]])

        selector = SemanticSelector(embedding_provider=mock_provider, seed=42)

        selected = selector.select(ds, k=2)

        assert len(selected) == 2
        assert mock_kmeans_instance.fit.called


def test_semantic_selector_backfill() -> None:
    """Test backfilling if clustering returns fewer unique points (unlikely but logic exists)."""

    examples = [TrainingExample(inputs={"q": i}, reference=i) for i in range(5)]
    ds = Dataset(examples)

    embeddings = [[0.0, 0.0] for _ in range(5)]
    mock_response = EmbeddingResponse(embeddings=embeddings, total_tokens=10, cost_usd=0.0, usage={})
    mock_provider = MagicMock(spec=EmbeddingProvider)
    mock_provider.embed.return_value = mock_response

    with patch("coreason_optimizer.strategies.selector.KMeans") as MockKMeans:
        kmeans_instance = MagicMock()
        MockKMeans.return_value = kmeans_instance

        # Assume 2 clusters requested, but everything assigned to cluster 0
        kmeans_instance.labels_ = np.array([0, 0, 0, 0, 0])
        kmeans_instance.cluster_centers_ = np.array([[0.0, 0.0], [1.0, 1.0]])

        selector = SemanticSelector(embedding_provider=mock_provider, seed=42)

        selected = selector.select(ds, k=2)

        # Should pick 1 from cluster 0, and backfill 1 random
        assert len(selected) == 2
        # Check they are unique
        assert selected[0] != selected[1]
