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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_optimizer.core.client import OpenAIClient, OpenAIClientAsync
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


@pytest.mark.asyncio
async def test_openai_client_async_initialization_with_key() -> None:
    with patch("coreason_optimizer.core.client.AsyncOpenAI") as MockAsyncOpenAI:
        # Mock close to be awaitable
        mock_instance = MockAsyncOpenAI.return_value
        mock_instance.close = AsyncMock()

        async with OpenAIClientAsync(api_key="test_key"):
            pass
        # Check call args of the mock class constructor
        call_args = MockAsyncOpenAI.call_args
        assert call_args.kwargs["api_key"] == "test_key"


@pytest.mark.asyncio
async def test_openai_client_async_initialization_with_env_var() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
        with patch("coreason_optimizer.core.client.AsyncOpenAI") as MockAsyncOpenAI:
            OpenAIClientAsync()
            assert MockAsyncOpenAI.call_args.kwargs["api_key"] == "env_key"


@pytest.mark.asyncio
async def test_openai_client_async_generate(mock_openai_response: MagicMock) -> None:
    # Fix: AsyncMock for nested calls
    # Remove spec=AsyncOpenAI to allow dynamic attributes in newer openai versions
    mock_client = AsyncMock()
    # create needs to be an async mock that returns the response
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    client = OpenAIClientAsync(client=mock_client)

    messages = [{"role": "user", "content": "Hello"}]
    response = await client.generate(messages, model="gpt-4o", temperature=0.5)

    assert isinstance(response, LLMResponse)
    assert response.content == "Test response"
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 50
    assert response.usage.total_tokens == 150
    assert response.usage.cost_usd == pytest.approx(0.00125)

    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
    )


def test_openai_client_sync_facade(mock_openai_response: MagicMock) -> None:
    """Test the synchronous facade wrapping the async client."""
    mock_client = AsyncMock()
    # Ensure create is awaitable
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    # Ensure close is awaitable because sync facade's __exit__ calls async __aexit__ which awaits client.close()
    mock_client.close = AsyncMock()

    # Inject the mocked async client
    with OpenAIClient(client=mock_client) as client:
        messages = [{"role": "user", "content": "Hello"}]
        response = client.generate(messages, model="gpt-4o", temperature=0.5)

    assert isinstance(response, LLMResponse)
    assert response.content == "Test response"
    assert response.usage.cost_usd == pytest.approx(0.00125)


@pytest.mark.asyncio
async def test_openai_client_async_generate_unknown_model(mock_openai_response: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    client = OpenAIClientAsync(client=mock_client)

    messages = [{"role": "user", "content": "Hello"}]
    response = await client.generate(messages, model="unknown-model")

    assert response.usage.cost_usd == 0.0


@pytest.mark.asyncio
async def test_openai_client_async_generate_no_usage(mock_openai_response: MagicMock) -> None:
    mock_openai_response.usage = None
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    client = OpenAIClientAsync(client=mock_client)

    response = await client.generate([{"role": "user", "content": "hi"}])
    assert response.usage.total_tokens == 0
    assert response.usage.cost_usd == 0.0


@pytest.mark.asyncio
async def test_openai_client_async_generate_failure() -> None:
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    client = OpenAIClientAsync(client=mock_client)

    with pytest.raises(Exception, match="API Error"):
        await client.generate([{"role": "user", "content": "Fail"}])
