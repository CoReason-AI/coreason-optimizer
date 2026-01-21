from unittest.mock import AsyncMock, patch

from coreason_optimizer.core.client import OpenAIClient, OpenAIEmbeddingClient


def test_openai_client_context_manager() -> None:
    with patch("coreason_optimizer.core.client.OpenAIClientAsync") as MockAsync:
        mock_async_instance = MockAsync.return_value
        mock_async_instance.__aexit__ = AsyncMock()

        with OpenAIClient(api_key="test") as client:
            assert client is not None

        mock_async_instance.__aexit__.assert_awaited_once()


def test_openai_embedding_client_context_manager() -> None:
    with patch("coreason_optimizer.core.client.OpenAIEmbeddingClientAsync") as MockAsync:
        mock_async_instance = MockAsync.return_value
        mock_async_instance.__aexit__ = AsyncMock()

        with OpenAIEmbeddingClient(api_key="test") as client:
            assert client is not None

        mock_async_instance.__aexit__.assert_awaited_once()
