from unittest.mock import AsyncMock

import pytest

from coreason_optimizer.core.interfaces import EmbeddingResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import RandomSelectorAsync, SemanticSelectorAsync


@pytest.fixture
def mock_dataset() -> Dataset:
    return Dataset(
        [
            TrainingExample(inputs={"q": "1"}, reference="A"),
            TrainingExample(inputs={"q": "2"}, reference="B"),
            TrainingExample(inputs={"q": "3"}, reference="C"),
        ]
    )


@pytest.mark.asyncio
async def test_random_selector_async(mock_dataset: Dataset) -> None:
    selector = RandomSelectorAsync(seed=42)
    selected = await selector.select(mock_dataset, k=2)
    assert len(selected) == 2

    # Test k >= len
    selected_all = await selector.select(mock_dataset, k=10)
    assert len(selected_all) == 3


@pytest.mark.asyncio
async def test_semantic_selector_async(mock_dataset: Dataset) -> None:
    mock_provider = AsyncMock()
    # Mock embedding: 3 examples, dim 2
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    mock_provider.embed.return_value = EmbeddingResponse(embeddings=embeddings, usage=UsageStats())

    selector = SemanticSelectorAsync(embedding_provider=mock_provider, seed=42)
    selected = await selector.select(mock_dataset, k=2)

    assert len(selected) == 2
    mock_provider.embed.assert_awaited_once()

    # Check fallback logic
    selected_k3 = await selector.select(mock_dataset, k=3)
    assert len(selected_k3) == 3


@pytest.mark.asyncio
async def test_semantic_selector_empty(mock_dataset: Dataset) -> None:
    mock_provider = AsyncMock()
    selector = SemanticSelectorAsync(embedding_provider=mock_provider)

    empty_ds = Dataset([])
    selected = await selector.select(empty_ds, k=2)
    assert len(selected) == 0
