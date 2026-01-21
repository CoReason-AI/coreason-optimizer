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

from coreason_optimizer.core.client import OpenAIClient
from coreason_optimizer.core.interfaces import LLMResponse


@pytest.fixture
def mock_openai_response() -> MagicMock:
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Test response"
    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150
    mock_response.usage = mock_usage

    return mock_response


def test_openai_client_initialization_with_key() -> None:
    # We need to patch OpenAIClientAsync now since OpenAIClient wraps it
    with patch("coreason_optimizer.core.client.OpenAIClientAsync") as MockAsync:
        client = OpenAIClient(api_key="test_key")
        MockAsync.assert_called_once_with(api_key="test_key")
        assert client._async_client == MockAsync.return_value


def test_openai_client_initialization_with_env_var() -> None:
    # This tests the wrapper logic passing args to the inner client
    # OpenAIClientAsync will handle the env var check
    with patch("coreason_optimizer.core.client.OpenAIClientAsync") as MockAsync:
        _ = OpenAIClient()
        MockAsync.assert_called_once_with(api_key=None)


def test_openai_client_generate(mock_openai_response: MagicMock) -> None:
    # Need to mock the Async client inside
    client = OpenAIClient(api_key="test")
    # Mock the internal async client
    client._async_client = AsyncMock()
    client._async_client.__aenter__.return_value = client._async_client

    # The return value from generate should be LLMResponse
    # We need to construct a real LLMResponse or a mock with attributes
    # The original test checked response.content, etc.
    # The wrapper returns what async client returns.

    # We can mock the return of async client generate
    async_response = LLMResponse(
        content="Test response",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost_usd": 0.00125},
    )
    client._async_client.generate.return_value = async_response

    messages = [{"role": "user", "content": "Hello"}]
    response = client.generate(messages, model="gpt-4o", temperature=0.5)

    assert isinstance(response, LLMResponse)
    assert response.content == "Test response"
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 50
    assert response.usage.total_tokens == 150
    assert response.usage.cost_usd == pytest.approx(0.00125)

    client._async_client.generate.assert_awaited_once_with(messages, "gpt-4o", 0.5)


def test_openai_client_generate_failure() -> None:
    client = OpenAIClient(api_key="test")
    client._async_client = AsyncMock()
    client._async_client.__aenter__.return_value = client._async_client
    client._async_client.generate.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        client.generate([{"role": "user", "content": "Fail"}])
