# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_optimizer.core.client import BudgetAwareEmbeddingProvider, OpenAIEmbeddingClient
from coreason_optimizer.core.interfaces import EmbeddingResponse


def test_embed_success() -> None:
    client = OpenAIEmbeddingClient(api_key="test")
    client._async_client = AsyncMock()
    client._async_client.__aenter__.return_value = client._async_client

    # Mock response
    mock_resp = EmbeddingResponse(embeddings=[[0.1, 0.2], [0.3, 0.4]], usage={"prompt_tokens": 10, "cost_usd": 0.0001})
    client._async_client.embed.return_value = mock_resp

    response = client.embed(["a", "b"])

    assert len(response.embeddings) == 2
    assert response.embeddings[0] == [0.1, 0.2]
    assert response.usage.prompt_tokens == 10
    assert response.usage.cost_usd > 0


def test_budget_aware_provider() -> None:
    # Need to pass an OpenAIEmbeddingClient because we enforce type check in refactor
    client = OpenAIEmbeddingClient(api_key="test")
    client._async_client = AsyncMock()

    # Mock usage
    usage = MagicMock(prompt_tokens=10, cost_usd=0.01)
    mock_resp = MagicMock(embeddings=[[1.0]], usage=usage)

    # We are testing the Sync Facade, so we need to mock the async methods
    # BUT BudgetAwareEmbeddingProvider wraps the async client directly
    # self._async = BudgetAwareEmbeddingProviderAsync(provider._async_client, ...)

    # So we need to mock provider._async_client.embed
    client._async_client.embed.return_value = mock_resp

    budget_manager = MagicMock()

    wrapper = BudgetAwareEmbeddingProvider(provider=client, budget_manager=budget_manager)
    wrapper.embed(["a"])

    budget_manager.check_budget.assert_called_once()
    # The async wrapper calls consume
    # Wait, wrapper.embed calls anyio.run(self._async.embed)
    # self._async is BudgetAwareEmbeddingProviderAsync
    # BudgetAwareEmbeddingProviderAsync calls manager.consume

    budget_manager.consume.assert_called_with(usage)


def test_embed_error() -> None:
    client = OpenAIEmbeddingClient(api_key="test")
    client._async_client = AsyncMock()
    client._async_client.__aenter__.return_value = client._async_client
    client._async_client.embed.side_effect = RuntimeError("API Error")

    with pytest.raises(RuntimeError):
        client.embed(["a"])


def test_init_default() -> None:
    # Test initialization
    # Now that we use OpenAIEmbeddingClientAsync, checks happen there.
    # We mock it to verify call args.

    with patch("coreason_optimizer.core.client.OpenAIEmbeddingClientAsync") as MockAsync:
        OpenAIEmbeddingClient(api_key="dummy")
        MockAsync.assert_called_with(api_key="dummy")


def test_embed_large_batch() -> None:
    # This logic moved to OpenAIEmbeddingClientAsync.
    # We should test it there or assume unit tests cover it.
    # Here we just verify the call is passed through.

    client = OpenAIEmbeddingClient(api_key="test")
    client._async_client = AsyncMock()
    client._async_client.__aenter__.return_value = client._async_client

    client.embed(["a"] * 505)

    client._async_client.embed.assert_awaited_once()
    # Arguments verification
    args, _ = client._async_client.embed.call_args
    assert len(args[0]) == 505
