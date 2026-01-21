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
LLM Client implementations for interacting with OpenAI API.

This module provides clients for generating text and embeddings,
along with wrappers for budget tracking.
"""

from types import TracebackType
from typing import Any, Optional, cast

import anyio
from openai import OpenAI

# Re-export pricing and cost calculation
from coreason_optimizer.core.async_client import (  # noqa: F401
    PRICING,
    BudgetAwareEmbeddingProviderAsync,
    BudgetAwareLLMClientAsync,
    OpenAIClientAsync,
    OpenAIEmbeddingClientAsync,
    calculate_openai_cost,
)
from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.core.interfaces import (
    EmbeddingProvider,
    EmbeddingResponse,
    LLMClient,
    LLMResponse,
)
from coreason_optimizer.utils.adapters import SyncToAsyncEmbeddingProviderAdapter, SyncToAsyncLLMClientAdapter


class OpenAIClient:
    """
    Sync Facade for OpenAIClientAsync.
    """

    def __init__(self, api_key: Optional[str] = None, client: Optional[OpenAI] = None):
        """
        Initialize the OpenAIClient (Sync Facade).
        """
        self._async_client = OpenAIClientAsync(api_key=api_key)

    def __enter__(self) -> "OpenAIClient":
        # We don't necessarily need to "enter" the async client here because
        # OpenAIClientAsync.__aenter__ is just `return self`.
        # However, for correctness/consistency, we could.
        # But we are in sync world. We can't await.
        # Since __aenter__ is trivial, we can skip it or run it.
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # We MUST run __aexit__ to close the client (session).
        async def _cleanup() -> None:
            await self._async_client.__aexit__(exc_type, exc_val, exc_tb)

        anyio.run(_cleanup)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the OpenAI LLM (Sync wrapper).
        """

        async def _run() -> LLMResponse:
            # We assume the client is managed by the context manager or self-managed.
            # If used as context manager, we just call generate.
            # If not, we might be leaking if we don't close.
            # But here we just forward the call.
            return await self._async_client.generate(messages, model, temperature, **kwargs)

        return cast(LLMResponse, anyio.run(_run))


class BudgetAwareLLMClient:
    """Sync Facade for BudgetAwareLLMClientAsync."""

    def __init__(self, client: LLMClient, budget_manager: BudgetManager):
        if hasattr(client, "_async_client"):
            async_client = client._async_client
        else:
            async_client = SyncToAsyncLLMClientAdapter(client)

        self._async = BudgetAwareLLMClientAsync(async_client, budget_manager)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        async def _run() -> LLMResponse:
            return await self._async.generate(messages, model, temperature, **kwargs)

        return cast(LLMResponse, anyio.run(_run))


class OpenAIEmbeddingClient:
    """Sync Facade for OpenAIEmbeddingClientAsync."""

    def __init__(self, api_key: Optional[str] = None, client: Optional[OpenAI] = None):
        self._async_client = OpenAIEmbeddingClientAsync(api_key=api_key)

    def __enter__(self) -> "OpenAIEmbeddingClient":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        async def _cleanup() -> None:
            await self._async_client.__aexit__(exc_type, exc_val, exc_tb)

        anyio.run(_cleanup)

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        async def _run() -> EmbeddingResponse:
            return await self._async_client.embed(texts, model)

        return cast(EmbeddingResponse, anyio.run(_run))


class BudgetAwareEmbeddingProvider:
    """Sync Facade for BudgetAwareEmbeddingProviderAsync."""

    def __init__(self, provider: EmbeddingProvider, budget_manager: BudgetManager):
        if hasattr(provider, "_async_client"):
            async_provider = provider._async_client
        else:
            async_provider = SyncToAsyncEmbeddingProviderAdapter(provider)

        self._async = BudgetAwareEmbeddingProviderAsync(async_provider, budget_manager)

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        async def _run() -> EmbeddingResponse:
            return await self._async.embed(texts, model)

        return cast(EmbeddingResponse, anyio.run(_run))
