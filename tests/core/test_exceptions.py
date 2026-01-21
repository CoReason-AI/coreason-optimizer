# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import os
from unittest.mock import AsyncMock

import pytest

# Mock env before import
os.environ["OPENAI_API_KEY"] = "sk-test-key"

from coreason_optimizer.core.budget import BudgetExceededError, BudgetManager
from coreason_optimizer.core.client import BudgetAwareLLMClient, OpenAIClient


def test_budget_exceeded_error_propagation() -> None:
    # Test that BudgetExceededError propagates through the Sync Facade

    # Setup
    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value = mock_async_client

    # Create Sync Facade
    client = OpenAIClient(api_key="test")
    client._async_client = mock_async_client

    # Budget Manager with low budget (but must be positive)
    budget_manager = BudgetManager(budget_limit_usd=0.01)

    # Manually exhaust budget
    budget_manager.total_cost_usd = 1.0

    # Wrap with BudgetAware
    aware_client = BudgetAwareLLMClient(client, budget_manager)

    # generate should raise BudgetExceededError synchronously
    with pytest.raises(BudgetExceededError):
        aware_client.generate([{"role": "user", "content": "hi"}])


def test_exception_propagation_through_facade() -> None:
    # Test generic exception
    mock_async_client = AsyncMock()
    # Ensure generate is mocked
    mock_async_client.generate.side_effect = ValueError("Custom Error")

    client = OpenAIClient(api_key="test")
    client._async_client = mock_async_client

    with pytest.raises(ValueError, match="Custom Error"):
        client.generate([])
