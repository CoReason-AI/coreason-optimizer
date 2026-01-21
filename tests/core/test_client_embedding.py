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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from coreason_optimizer.core.client import BudgetAwareEmbeddingProvider, OpenAIEmbeddingClient


def test_embed_success() -> None:
    mock_client = MagicMock()
    # Mock response
    mock_response = MagicMock()
    mock_data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    mock_response.data = mock_data
    mock_response.usage.prompt_tokens = 10
    mock_client.embeddings.create.return_value = mock_response

    client = OpenAIEmbeddingClient(client=mock_client)
    response = client.embed(["a", "b"])

    assert len(response.embeddings) == 2
    assert response.embeddings[0] == [0.1, 0.2]
    assert response.usage.prompt_tokens == 10
    assert response.usage.cost_usd > 0  # Should be calculated


def test_budget_aware_provider() -> None:
    mock_provider = MagicMock()
    # Mock usage
    usage = MagicMock(prompt_tokens=10, cost_usd=0.01)
    mock_provider.embed.return_value = MagicMock(embeddings=[[1.0]], usage=usage)

    budget_manager = MagicMock()

    wrapper = BudgetAwareEmbeddingProvider(provider=mock_provider, budget_manager=budget_manager)
    wrapper.embed(["a"])

    budget_manager.check_budget.assert_called_once()
    budget_manager.consume.assert_called_with(usage)


def test_embed_error() -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("API Error")

    client = OpenAIEmbeddingClient(client=mock_client)
    with pytest.raises(RuntimeError):
        client.embed(["a"])


def test_init_default() -> None:
    # Test initialization without client (reads env var, assumes mock/env)
    # If OPENAI_API_KEY is present, it succeeds. If not, it raises.

    # Force failure
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(OpenAIError):
            OpenAIEmbeddingClient()

    # Force success
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
        c = OpenAIEmbeddingClient()
        assert c.client is not None

    # If we pass client, it works
    c = OpenAIEmbeddingClient(client=MagicMock())
    assert c.client is not None


def test_embed_large_batch() -> None:
    # Test that client batches requests if input is larger than batch_size (500)
    mock_client = MagicMock()
    # We want 505 items.
    # 1st call: 500 items. 2nd call: 5 items.

    # Setup response side_effect
    def side_effect(input: list[str], model: str) -> Any:
        count = len(input)
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.0] * 2) for _ in range(count)]
        resp.usage.prompt_tokens = count
        return resp

    mock_client.embeddings.create.side_effect = side_effect

    client = OpenAIEmbeddingClient(client=mock_client)

    # Generate 505 items
    inputs = [str(i) for i in range(505)]
    response = client.embed(inputs)

    assert len(response.embeddings) == 505
    assert response.usage.prompt_tokens == 505
    assert mock_client.embeddings.create.call_count == 2
