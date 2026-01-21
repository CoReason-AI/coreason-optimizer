from unittest.mock import MagicMock

from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.core.client import BudgetAwareEmbeddingProvider
from coreason_optimizer.core.interfaces import EmbeddingProvider, EmbeddingResponse, UsageStats
from coreason_optimizer.utils.adapters import SyncToAsyncEmbeddingProviderAdapter


def test_budget_aware_embedding_provider_adapter_fallback() -> None:
    # Test that a generic provider is wrapped with Adapter
    mock_provider = MagicMock(spec=EmbeddingProvider)
    # Ensure no _async_client attribute
    del mock_provider._async_client

    budget_manager = MagicMock(spec=BudgetManager)

    wrapper = BudgetAwareEmbeddingProvider(mock_provider, budget_manager)

    # Access inner async wrapper to verify it is Adapter
    # wrapper._async is BudgetAwareEmbeddingProviderAsync
    # wrapper._async.provider should be SyncToAsyncEmbeddingProviderAdapter
    assert isinstance(wrapper._async.provider, SyncToAsyncEmbeddingProviderAdapter)

    # Test execution
    mock_provider.embed.return_value = EmbeddingResponse(embeddings=[[0.1]], usage=UsageStats())

    res = wrapper.embed(["test"])
    assert len(res.embeddings) == 1
    mock_provider.embed.assert_called_once()
