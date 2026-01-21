from unittest.mock import AsyncMock, patch

import pytest

from coreason_optimizer.core.async_client import OpenAIClientAsync


@pytest.mark.asyncio
async def test_async_client_default_init() -> None:
    # Mock AsyncOpenAI constructor to prevent real network call
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_instance = MockOpenAI.return_value
        mock_instance.close = AsyncMock()

        async with OpenAIClientAsync(api_key="test") as client:
            assert client._internal_client is True
            assert client.client is mock_instance

        mock_instance.close.assert_awaited_once()
