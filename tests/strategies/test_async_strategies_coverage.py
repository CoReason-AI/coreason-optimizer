from unittest.mock import AsyncMock, MagicMock

import pytest

from coreason_optimizer.core.interfaces import EmbeddingResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.mutator import IdentityMutatorAsync
from coreason_optimizer.strategies.selector import SemanticSelector


@pytest.mark.asyncio
async def test_identity_mutator_async() -> None:
    client = AsyncMock()
    mutator = IdentityMutatorAsync(client)
    instruction = "test"
    result = await mutator.mutate(instruction)
    assert result == instruction


def test_semantic_selector_sync() -> None:
    mock_provider = MagicMock()
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    mock_provider.embed.return_value = EmbeddingResponse(embeddings=embeddings, usage=UsageStats())

    selector = SemanticSelector(embedding_provider=mock_provider, seed=42)
    dataset = Dataset(
        [
            TrainingExample(inputs={"q": "1"}, reference="A"),
            TrainingExample(inputs={"q": "2"}, reference="B"),
            TrainingExample(inputs={"q": "3"}, reference="C"),
        ]
    )

    selected = selector.select(dataset, k=2)
    assert len(selected) == 2
    mock_provider.embed.assert_called_once()
