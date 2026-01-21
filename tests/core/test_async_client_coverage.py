from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_optimizer.core.async_client import OpenAIClientAsync, OpenAIEmbeddingClientAsync


@pytest.mark.asyncio
async def test_async_client_context_manager_internal() -> None:
    # Test OpenAIClientAsync context manager with internal client (default)
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_instance = MockOpenAI.return_value
        mock_instance.close = AsyncMock()

        async with OpenAIClientAsync(api_key="test") as client:
            assert client._internal_client is True

        mock_instance.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_embedding_client_context_manager_internal() -> None:
    # Test OpenAIEmbeddingClientAsync context manager with internal client
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_instance = MockOpenAI.return_value
        mock_instance.close = AsyncMock()

        async with OpenAIEmbeddingClientAsync(api_key="test") as client:
            assert client._internal_client is True

        mock_instance.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_embedding_client_context_manager_external() -> None:
    # Test OpenAIEmbeddingClientAsync context manager with external client
    mock_instance = AsyncMock()
    mock_instance.close = AsyncMock()

    async with OpenAIEmbeddingClientAsync(client=mock_instance) as client:
        assert client._internal_client is False

    # Should NOT be closed
    mock_instance.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_client_generate_usage_variations() -> None:
    # Test usage stats variations (present vs None)
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_instance = MockOpenAI.return_value

        # 1. With usage
        # We must set create as AsyncMock
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="content"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )
        )

        client = OpenAIClientAsync(api_key="test")
        resp = await client.generate([])
        assert resp.usage.total_tokens == 20
        assert resp.usage.cost_usd > 0

        # 2. Without usage (None)
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="content"))], usage=None)
        )

        resp_no_usage = await client.generate([])
        assert resp_no_usage.usage.total_tokens == 0
        assert resp_no_usage.usage.cost_usd == 0.0
