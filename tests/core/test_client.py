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
from openai import AsyncOpenAI

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
async def test_openai_client_initialization_with_key() -> None:
    # Patch AsyncOpenAI instead of OpenAI
    with patch("coreason_optimizer.core.client.AsyncOpenAI") as MockOpenAI:
        async with OpenAIClientAsync(api_key="test_key"):
            # Check that api_key was passed. Note that http_client is also passed.
            call_args = MockOpenAI.call_args
            assert call_args.kwargs["api_key"] == "test_key"
            assert "http_client" in call_args.kwargs
            # We don't need to check client.client == MockOpenAI.return_value strictly if we trust init logic


@pytest.mark.asyncio
async def test_openai_client_initialization_with_env_var() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
        with patch("coreason_optimizer.core.client.AsyncOpenAI") as MockOpenAI:
            async with OpenAIClientAsync() as _:
                call_args = MockOpenAI.call_args
                assert call_args.kwargs["api_key"] == "env_key"


@pytest.mark.asyncio
async def test_openai_client_generate(mock_openai_response: MagicMock) -> None:
    # AsyncMock for the client
    mock_client = AsyncMock(spec=AsyncOpenAI)
    # The chain is client.chat.completions.create
    # Need to make the create method an AsyncMock or return a value that can be awaited if using MagicMock on top level?
    # Using AsyncMock automatically makes methods awaitable unless return_value is set to non-awaitable?
    # Actually, client.chat.completions.create needs to be async.
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    with patch("coreason_optimizer.core.client.AsyncOpenAI", return_value=mock_client):
        async with OpenAIClientAsync() as client:
            messages = [{"role": "user", "content": "Hello"}]
            response = await client.generate(messages, model="gpt-4o", temperature=0.5)

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.usage.prompt_tokens == 100
            assert response.usage.completion_tokens == 50
            assert response.usage.total_tokens == 150

            # Cost calculation check for gpt-4o
            # Input: 100 * 5.00 / 1M = 0.0005
            # Output: 50 * 15.00 / 1M = 0.00075
            # Total: 0.00125
            assert response.usage.cost_usd == pytest.approx(0.00125)

            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o",
                messages=messages,
                temperature=0.5,
            )


@pytest.mark.asyncio
async def test_openai_client_cost_calculation_mini(mock_openai_response: MagicMock) -> None:
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    with patch("coreason_optimizer.core.client.AsyncOpenAI", return_value=mock_client):
        async with OpenAIClientAsync() as client:
            # gpt-4o-mini pricing: input 0.15, output 0.60
            # Usage: 100 input, 50 output
            # Input cost: 100 * 0.15 / 1M = 0.000015
            # Output cost: 50 * 0.60 / 1M = 0.000030
            # Total: 0.000045

            response = await client.generate([{"role": "user", "content": "hi"}], model="gpt-4o-mini")
            assert response.usage.cost_usd == pytest.approx(0.000045)


@pytest.mark.asyncio
async def test_openai_client_generate_unknown_model(mock_openai_response: MagicMock) -> None:
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    with patch("coreason_optimizer.core.client.AsyncOpenAI", return_value=mock_client):
        async with OpenAIClientAsync() as client:
            messages = [{"role": "user", "content": "Hello"}]
            response = await client.generate(messages, model="unknown-model")

            assert response.usage.cost_usd == 0.0


@pytest.mark.asyncio
async def test_openai_client_generate_no_usage(mock_openai_response: MagicMock) -> None:
    mock_openai_response.usage = None
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    with patch("coreason_optimizer.core.client.AsyncOpenAI", return_value=mock_client):
        async with OpenAIClientAsync() as client:
            response = await client.generate([{"role": "user", "content": "hi"}])
            assert response.usage.total_tokens == 0
            assert response.usage.cost_usd == 0.0


@pytest.mark.asyncio
async def test_openai_client_generate_failure() -> None:
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    with patch("coreason_optimizer.core.client.AsyncOpenAI", return_value=mock_client):
        async with OpenAIClientAsync() as client:
            with pytest.raises(Exception, match="API Error"):
                await client.generate([{"role": "user", "content": "Fail"}])


def test_openai_client_sync_facade(mock_openai_response: MagicMock) -> None:
    """Test that the Sync Facade works (wraps async correctly)."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    # We patch AsyncOpenAI used inside OpenAIClientAsync, which is used by OpenAIClient
    with patch("coreason_optimizer.core.client.AsyncOpenAI", return_value=mock_client):
        with OpenAIClient() as client:
            messages = [{"role": "user", "content": "Hello"}]
            response = client.generate(messages, model="gpt-4o", temperature=0.5)

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
