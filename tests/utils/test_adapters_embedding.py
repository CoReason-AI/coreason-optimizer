from unittest.mock import MagicMock

import pytest

from coreason_optimizer.core.interfaces import EmbeddingResponse, UsageStats
from coreason_optimizer.utils.adapters import SyncToAsyncEmbeddingProviderAdapter


@pytest.mark.asyncio
async def test_embedding_adapter_works() -> None:
    mock_sync_provider = MagicMock()
    mock_sync_provider.embed.return_value = EmbeddingResponse(embeddings=[[0.1]], usage=UsageStats())

    adapter = SyncToAsyncEmbeddingProviderAdapter(mock_sync_provider)
    result = await adapter.embed(["test"])

    assert len(result.embeddings) == 1
    mock_sync_provider.embed.assert_called_once_with(["test"], model=None)
