import os
from unittest.mock import AsyncMock

import pytest

# Mock env before import if possible or just rely on passing key
os.environ["OPENAI_API_KEY"] = "sk-test-key"

from coreason_optimizer.core.async_client import OpenAIClientAsync
from coreason_optimizer.core.client import OpenAIClient
from coreason_optimizer.core.interfaces import LLMResponse, UsageStats


@pytest.mark.asyncio
async def test_async_client_lifecycle() -> None:
    # Mocking internal client behavior would be complex without real network,
    # but we can verify the context manager structure.

    # We can inject a mock AsyncOpenAI client
    mock_openai = AsyncMock()
    # Ensure close is a coroutine mock
    mock_openai.close = AsyncMock()

    async with OpenAIClientAsync(client=mock_openai) as client:
        assert client.client is mock_openai
        # We can't easily mock generate without detailed response structure mocking
        # but we tested the lifecycle entrance
        pass

    # The client was passed internally, so _internal_client is False, so close should NOT be called.
    # Logic in async_client.py:
    # self._internal_client = client is None
    # if self._internal_client: await self.client.close()

    mock_openai.close.assert_not_awaited()

    # Test with internal client (no client passed)
    # We need to mock AsyncOpenAI constructor but it is hard imported.
    # Instead we check logic:

    client2 = OpenAIClientAsync(client=None, api_key="sk-test")
    # Mock the client attribute manually
    client2.client = mock_openai
    # Set internal flag manually if needed but __init__ sets it.
    # client=None passed to init sets _internal_client=True

    async with client2:
        pass

    mock_openai.close.assert_awaited_once()


def test_sync_client_facade() -> None:
    # Test that Sync Facade calls Async Client via anyio.run

    # We need to mock the internal async client of the sync client
    mock_async_client = AsyncMock(spec=OpenAIClientAsync)
    # NOTE: In our refactored implementation, OpenAIClient.generate does NOT use context manager anymore
    # It calls await self._async_client.generate directly.
    # Context manager is only used if user does `with OpenAIClient()`.

    expected_response = LLMResponse(content="test", usage=UsageStats())
    mock_async_client.generate = AsyncMock(return_value=expected_response)

    # We need to bypass __init__ or provide api_key
    client = OpenAIClient(api_key="sk-test")
    # Inject mock
    client._async_client = mock_async_client

    # Call sync method
    response = client.generate(messages=[])

    assert response == expected_response
    mock_async_client.generate.assert_awaited_once()

    # Verify __enter__ usage if we use context manager
    with client:
        pass
        # __enter__ does nothing

    # But __exit__ should call async exit
    # We can't easily test __exit__ calling anyio.run here because we are mocking the internal client
    # and mocking anyio.run is tricky if we are already in sync test.
    # But checking generate logic is sufficient for basic functionality.
