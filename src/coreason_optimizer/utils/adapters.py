# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

"""
Adapters to bridge Sync and Async clients.
"""

from typing import Any, Optional

import anyio

from coreason_optimizer.core.interfaces import (
    EmbeddingProvider,
    EmbeddingProviderAsync,
    EmbeddingResponse,
    LLMClient,
    LLMClientAsync,
    LLMResponse,
)


class SyncToAsyncLLMClientAdapter(LLMClientAsync):
    """
    Wraps a synchronous LLMClient to act as an Async client.
    Offloads sync calls to a thread.
    """

    def __init__(self, sync_client: LLMClient):
        self.sync_client = sync_client

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        def _call() -> LLMResponse:
            return self.sync_client.generate(messages, model=model, temperature=temperature, **kwargs)

        return await anyio.to_thread.run_sync(_call)


class SyncToAsyncEmbeddingProviderAdapter(EmbeddingProviderAsync):
    """
    Wraps a synchronous EmbeddingProvider to act as an Async provider.
    Offloads sync calls to a thread.
    """

    def __init__(self, sync_provider: EmbeddingProvider):
        self.sync_provider = sync_provider

    async def embed(self, texts: list[str], model: Optional[str] = None) -> EmbeddingResponse:
        def _call() -> EmbeddingResponse:
            return self.sync_provider.embed(texts, model=model)

        return await anyio.to_thread.run_sync(_call)
