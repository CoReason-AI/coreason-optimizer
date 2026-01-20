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
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from coreason_optimizer.core.client import OpenAIEmbeddingClient


def test_embed_success() -> None:
    mock_client = MagicMock()
    # Mock response
    mock_response = MagicMock()
    mock_data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    mock_response.data = mock_data
    mock_response.usage.prompt_tokens = 10
    mock_client.embeddings.create.return_value = mock_response

    budget_manager = MagicMock()

    client = OpenAIEmbeddingClient(client=mock_client, budget_manager=budget_manager)
    embeddings = client.embed(["a", "b"])

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2]

    # Check budget consumption
    budget_manager.consume.assert_called_once()
    usage = budget_manager.consume.call_args[0][0]
    assert usage.prompt_tokens == 10
    assert usage.cost_usd > 0


def test_embed_no_budget() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1])]
    mock_response.usage = None  # Handle missing usage
    mock_client.embeddings.create.return_value = mock_response

    client = OpenAIEmbeddingClient(client=mock_client)
    embeddings = client.embed(["a"])
    assert len(embeddings) == 1


def test_embed_error() -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("API Error")

    client = OpenAIEmbeddingClient(client=mock_client)
    with pytest.raises(RuntimeError):
        client.embed(["a"])


def test_init_default() -> None:
    # Test initialization without client (reads env var, assumes mock/env)
    # If OPENAI_API_KEY is present, it succeeds. If not, it raises.
    # We can patch os.environ to force failure or success.

    # Force failure
    with patch.dict(os.environ, {}, clear=True):
        # OpenAI() raises if no key provided and no env var
        with pytest.raises(OpenAIError):
            OpenAIEmbeddingClient()

    # Force success
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
        c = OpenAIEmbeddingClient()
        assert c.client is not None

    # If we pass client, it works
    c = OpenAIEmbeddingClient(client=MagicMock())
    assert c.client is not None
