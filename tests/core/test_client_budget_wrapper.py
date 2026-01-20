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
from unittest.mock import MagicMock

import pytest

from coreason_optimizer.core.budget import BudgetExceededError, BudgetManager
from coreason_optimizer.core.client import BudgetAwareLLMClient
from coreason_optimizer.core.interfaces import LLMClient, LLMResponse, UsageStats


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
