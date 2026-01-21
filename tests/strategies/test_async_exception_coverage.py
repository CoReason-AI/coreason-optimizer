from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import EmbeddingResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.bootstrap import BootstrapFewShot, BootstrapFewShotAsync
from coreason_optimizer.strategies.mipro import MiproOptimizer
from coreason_optimizer.strategies.selector import SemanticSelector
from coreason_optimizer.utils.adapters import SyncToAsyncEmbeddingProviderAdapter, SyncToAsyncLLMClientAdapter


@pytest.mark.asyncio
async def test_bootstrap_evaluate_one_exception() -> None:
    # Test that _evaluate_one swallows generic exceptions
    mock_client = AsyncMock()

    agent = MagicMock()
    agent.system_prompt = "system prompt"

    valset = [TrainingExample(inputs={"q": "1"}, reference="A")]

    # Side effect: 1st call success (mining), 2nd call fail (validation)
    success_resp = MagicMock(content="prediction", usage=UsageStats())

    # We can use an iterator side_effect
    mock_client.generate.side_effect = [
        success_resp,  # Mining example 1
        ValueError("Validation Error"),  # Validation example 1
    ]

    metric = MagicMock(return_value=1.0)

    bs = BootstrapFewShotAsync(mock_client, metric, OptimizerConfig(max_bootstrapped_demos=1))

    manifest = await bs.compile(agent, [TrainingExample(inputs={"q": "1"}, reference="A")], valset)

    assert manifest.performance_metric == 0.0


def test_sync_facade_uses_adapter_bad_client() -> None:
    # Test that generic client is wrapped in Adapter
    mock_client = MagicMock()  # Not OpenAIClient
    del mock_client._async_client  # Ensure no attribute

    bs = BootstrapFewShot(mock_client, MagicMock(), OptimizerConfig())
    # Verify adapter usage. llm_client is BudgetAwareLLMClientAsync. .client is the inner client.
    assert isinstance(bs._async.llm_client.client, SyncToAsyncLLMClientAdapter)

    mo = MiproOptimizer(mock_client, MagicMock(), OptimizerConfig())
    assert isinstance(mo._async.llm_client.client, SyncToAsyncLLMClientAdapter)


def test_mipro_sync_uses_adapter_bad_embedding() -> None:
    mock_client = MagicMock()
    mock_client._async_client = AsyncMock()

    mock_embed = MagicMock()
    del mock_embed._async_client

    # Enable semantic selection
    config = OptimizerConfig(selector_type="semantic", embedding_model="emb")
    mo = MiproOptimizer(mock_client, MagicMock(), config, embedding_provider=mock_embed)

    # Verify adapter
    # mo._async.selector.embedding_provider.provider
    # Need to cast to access dynamic attributes safely for mypy or ignore
    # But BaseSelectorAsync doesn't have embedding_provider field in type hint?
    # It does in SemanticSelectorAsync implementation.

    selector = cast(Any, mo._async.selector)
    assert isinstance(selector.embedding_provider.provider, SyncToAsyncEmbeddingProviderAdapter)


def test_sync_semantic_selector_coverage() -> None:
    # Test Sync SemanticSelector fallback logic
    mock_provider = MagicMock()
    mock_provider.embed.return_value = EmbeddingResponse(embeddings=[[0.1, 0.2]], usage=UsageStats())

    selector = SemanticSelector(mock_provider, seed=42)
    ds = Dataset([TrainingExample(inputs={"q": "1"}, reference="A")])

    # Test short circuit
    res = selector.select(ds, k=5)
    assert len(res) == 1
    mock_provider.embed.assert_not_called()

    # Test clustering (k < len)
    # Need more examples
    ds_large = Dataset([TrainingExample(inputs={"q": str(i)}, reference="A") for i in range(5)])
    mock_provider.embed.return_value = EmbeddingResponse(embeddings=[[0.1] * 1536] * 5, usage=UsageStats())

    res = selector.select(ds_large, k=2)
    assert len(res) == 2
    mock_provider.embed.assert_called()
