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
from openai import AsyncOpenAI, OpenAIError

from coreason_optimizer.core.client import OpenAIClient


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
    # We now mock AsyncOpenAI because OpenAIClient uses OpenAIClientAsync internally
    mock_client = AsyncMock(spec=AsyncOpenAI)
    client = OpenAIClient(client=mock_client)

    with pytest.raises(ValueError, match="Streaming is not supported"):
        client.generate([], stream=True)


def test_empty_content_handled(mock_openai_response: MagicMock) -> None:
    mock_openai_response.choices[0].message.content = None
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    # Also need to mock close() because context manager might call it or we might need it if we used context manager
    # OpenAIClient facade doesn't strictly need close() for generate(), but __exit__ does.
    # Here we are just initializing and calling generate.

    client = OpenAIClient(client=mock_client)

    resp = client.generate([])
    assert resp.content == ""


def test_multiple_n_handled(mock_openai_response: MagicMock) -> None:
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    client = OpenAIClient(client=mock_client)

    # We pass n=2, response still has 1 choice mocked but usage reflects total.
    client.generate([], n=2)
    mock_client.chat.completions.create.assert_called_once()
    assert mock_client.chat.completions.create.call_args.kwargs["n"] == 2


def test_missing_api_key_raises_error() -> None:
    # Ensure environment is clean
    with patch.dict(os.environ, {}, clear=True):
        # OpenAIClient now initializes OpenAIClientAsync which initializes AsyncOpenAI
        # OpenAI 1.0+ checks for API key at instantiation
        with pytest.raises(OpenAIError):
            OpenAIClient()
