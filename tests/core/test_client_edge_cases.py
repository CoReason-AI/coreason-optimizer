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
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

from coreason_optimizer.core.client import OpenAIClient
from coreason_optimizer.core.interfaces import LLMResponse


@pytest.fixture
def mock_openai_response() -> MagicMock:
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Test response"
    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 10
    mock_usage.total_tokens = 20
    mock_response.usage = mock_usage

    return mock_response


def test_stream_raises_error(mock_openai_response: MagicMock) -> None:
    # Need to check async client validation
    with patch("coreason_optimizer.core.client.OpenAIClientAsync"):
        client = OpenAIClient(api_key="test")
        # Ensure generate is AsyncMock
        cast(Any, client._async_client).generate = AsyncMock()
        cast(Any, client._async_client).generate.side_effect = ValueError("Streaming is not supported")

        with pytest.raises(ValueError, match="Streaming is not supported"):
            client.generate([], stream=True)


def test_empty_content_handled(mock_openai_response: MagicMock) -> None:
    # Mock return from async client
    with patch("coreason_optimizer.core.client.OpenAIClientAsync"):
        client = OpenAIClient(api_key="test")

        # Async client should return LLMResponse with empty string
        cast(Any, client._async_client).generate = AsyncMock()
        cast(Any, client._async_client).generate.return_value = LLMResponse(
            content="",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost_usd": 0.0},
        )

        resp = client.generate([])
        assert resp.content == ""


def test_multiple_n_handled(mock_openai_response: MagicMock) -> None:
    with patch("coreason_optimizer.core.client.OpenAIClientAsync"):
        client = OpenAIClient(api_key="test")

        cast(Any, client._async_client).generate = AsyncMock()
        cast(Any, client._async_client).generate.return_value = LLMResponse(content="test", usage={})

        # We pass n=2
        client.generate([], n=2)

        cast(Any, client._async_client).generate.assert_awaited_once()
        call_args = cast(Any, client._async_client).generate.call_args
        assert call_args.kwargs["n"] == 2


def test_missing_api_key_raises_error() -> None:
    # Ensure environment is clean
    with patch.dict(os.environ, {}, clear=True):
        # OpenAIClient init calls OpenAIClientAsync init, which checks env var
        with pytest.raises(OpenAIError):
            OpenAIClient()
