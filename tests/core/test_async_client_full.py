from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from coreason_optimizer.core.async_client import OpenAIClientAsync, OpenAIEmbeddingClientAsync


@pytest.mark.asyncio
async def test_embedding_client_async_batching() -> None:
    # Test batching logic in OpenAIEmbeddingClientAsync.embed
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value

        # Setup response side effect for batching
        async def side_effect(input: Any, model: str | None) -> Any:
            count = len(input)
            resp = AsyncMock()
            resp.data = [AsyncMock(embedding=[0.1] * 1536) for _ in range(count)]
            resp.usage.prompt_tokens = count
            return resp

        mock_client.embeddings.create.side_effect = side_effect

        client = OpenAIEmbeddingClientAsync(api_key="test")

        # 505 items -> 2 batches (500 + 5)
        texts = ["t"] * 505
        response = await client.embed(texts)

        assert len(response.embeddings) == 505
        assert response.usage.prompt_tokens == 505
        # Use call_count for MagicMock (patched class)
        assert mock_client.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_llm_client_async_streaming_error() -> None:
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI"):
        client = OpenAIClientAsync(api_key="test")
        with pytest.raises(ValueError, match="Streaming is not supported"):
            await client.generate([], stream=True)


@pytest.mark.asyncio
async def test_llm_client_async_api_error() -> None:
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = RuntimeError("API Fail")

        client = OpenAIClientAsync(api_key="test")
        with pytest.raises(RuntimeError):
            await client.generate([])


@pytest.mark.asyncio
async def test_embedding_client_async_api_error() -> None:
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.embeddings.create.side_effect = RuntimeError("API Fail")

        client = OpenAIEmbeddingClientAsync(api_key="test")
        with pytest.raises(RuntimeError):
            await client.embed(["t"])
