# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_optimizer.core.budget import BudgetExceededError, BudgetManager
from coreason_optimizer.core.client import (
    BudgetAwareEmbeddingProvider,
    BudgetAwareEmbeddingProviderAsync,
    BudgetAwareLLMClient,
    BudgetAwareLLMClientAsync,
    OpenAIClient,
)
from coreason_optimizer.core.interfaces import (
    AsyncEmbeddingProvider,
    AsyncLLMClient,
    EmbeddingProvider,
    EmbeddingResponse,
    LLMClient,
    LLMResponse,
    UsageStats,
)


class MockLLMClient(LLMClient):
    """Mock LLM Client for testing."""

    def __init__(self, response_cost: float = 0.0):
        self.response_cost = response_cost

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        usage = UsageStats(cost_usd=self.response_cost)
        return LLMResponse(content="mock", usage=usage)


class MockLLMClientAsync(AsyncLLMClient):
    """Mock Async LLM Client for testing."""

    def __init__(self, response_cost: float = 0.0):
        self.response_cost = response_cost

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        usage = UsageStats(cost_usd=self.response_cost)
        return LLMResponse(content="mock", usage=usage)


def test_wrapper_delegates_and_tracks_cost() -> None:
    """Test that the wrapper calls inner client and updates budget."""
    inner = MockLLMClient(response_cost=1.0)
    budget = BudgetManager(budget_limit_usd=10.0)
    wrapper = BudgetAwareLLMClient(inner, budget)

    response = wrapper.generate([{"role": "user", "content": "hi"}])
    assert response.content == "mock"
    assert budget.total_cost_usd == 1.0

    wrapper.generate([{"role": "user", "content": "hi"}])
    assert budget.total_cost_usd == 2.0


@pytest.mark.asyncio
async def test_async_wrapper_delegates_and_tracks_cost() -> None:
    """Test that the async wrapper calls inner client and updates budget."""
    inner = MockLLMClientAsync(response_cost=1.0)
    budget = BudgetManager(budget_limit_usd=10.0)
    wrapper = BudgetAwareLLMClientAsync(inner, budget)

    response = await wrapper.generate([{"role": "user", "content": "hi"}])
    assert response.content == "mock"
    assert budget.total_cost_usd == 1.0


def test_wrapper_raises_budget_exceeded() -> None:
    """Test that the wrapper raises exception when budget exceeded."""
    inner = MockLLMClient(response_cost=6.0)
    budget = BudgetManager(budget_limit_usd=10.0)
    wrapper = BudgetAwareLLMClient(inner, budget)

    # First call: 6.0 <= 10.0. OK.
    wrapper.generate([])
    assert budget.total_cost_usd == 6.0

    # Second call: 12.0 > 10.0. Fail.
    with pytest.raises(BudgetExceededError):
        wrapper.generate([])


def test_wrapper_passes_args() -> None:
    """Test that arguments are passed to inner client."""
    mock_inner = MagicMock()
    mock_inner.generate.return_value = LLMResponse(content="", usage=UsageStats())

    budget = BudgetManager(10.0)
    wrapper = BudgetAwareLLMClient(mock_inner, budget)

    wrapper.generate(messages=[{"a": "b"}], model="gpt-test", temperature=0.7, extra="val")

    mock_inner.generate.assert_called_once_with(messages=[{"a": "b"}], model="gpt-test", temperature=0.7, extra="val")


def test_wrapper_blocks_call_if_already_exceeded() -> None:
    """Test that generate is blocked if budget is already exceeded."""
    mock_inner = MagicMock()
    budget = BudgetManager(5.0)
    wrapper = BudgetAwareLLMClient(mock_inner, budget)

    # Manually consume budget to exceed it
    try:
        budget.consume(UsageStats(cost_usd=6.0))
    except BudgetExceededError:
        pass  # Expected

    # Try generate
    with pytest.raises(BudgetExceededError):
        wrapper.generate([])

    # Ensure inner client was NOT called
    mock_inner.generate.assert_not_called()


def test_boundary_conditions() -> None:
    """Test exact budget match."""
    inner = MockLLMClient(response_cost=5.0)
    budget = BudgetManager(10.0)
    wrapper = BudgetAwareLLMClient(inner, budget)

    # 1. Spend exactly 5.0. Total 5.0 <= 10.0. OK.
    wrapper.generate([])
    assert budget.total_cost_usd == 5.0

    # 2. Spend another 5.0. Total 10.0 <= 10.0. OK. (Boundary)
    wrapper.generate([])
    assert budget.total_cost_usd == 10.0

    # 3. Spend 0.1 more. Total 10.1 > 10.0. Fail.
    inner.response_cost = 0.1
    with pytest.raises(BudgetExceededError):
        wrapper.generate([])


class MockEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        return EmbeddingResponse(embeddings=[], usage=UsageStats(cost_usd=1.0))


class MockEmbeddingProviderAsync(AsyncEmbeddingProvider):
    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        return EmbeddingResponse(embeddings=[], usage=UsageStats(cost_usd=1.0))


def test_embedding_wrapper_sync() -> None:
    inner = MockEmbeddingProvider()
    budget = BudgetManager(10.0)
    wrapper = BudgetAwareEmbeddingProvider(inner, budget)

    wrapper.embed(["test"])
    assert budget.total_cost_usd == 1.0


@pytest.mark.asyncio
async def test_embedding_wrapper_async() -> None:
    inner = MockEmbeddingProviderAsync()
    budget = BudgetManager(10.0)
    wrapper = BudgetAwareEmbeddingProviderAsync(inner, budget)

    await wrapper.embed(["test"])
    assert budget.total_cost_usd == 1.0


def test_openai_client_exit_exception() -> None:
    """Test that OpenAIClient.__exit__ propagates exceptions properly."""
    # We must patch AsyncOpenAI because OpenAIClient init will try to create one if not passed
    with patch("coreason_optimizer.core.client.AsyncOpenAI") as MockAsyncOpenAI:
        MockAsyncOpenAI.return_value = AsyncMock()
        client = OpenAIClient(api_key="test")

        # Mock the internal async client behavior
        mock_async = MagicMock()

        # Setup __aexit__ to raise
        async def mock_aexit(*args: Any) -> None:
            raise ValueError("Inner error")

        mock_async.__aexit__ = mock_aexit
        client._async = mock_async

        with pytest.raises(ValueError, match="Inner error"):
            with client:
                pass
