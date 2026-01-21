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

import os
from typing import Any, Optional

import anyio
import httpx
from openai import AsyncOpenAI

from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.core.interfaces import (
    AsyncEmbeddingProvider,
    AsyncLLMClient,
    EmbeddingProvider,
    EmbeddingResponse,
    LLMClient,
    LLMResponse,
    UsageStats,
)
from coreason_optimizer.utils.logger import logger

# Pricing per 1M tokens (approximate as of early 2025)
PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def calculate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of an OpenAI API call based on model and usage.

    Args:
        model: The model identifier (e.g., 'gpt-4o').
        input_tokens: Number of prompt tokens.
        output_tokens: Number of completion tokens.

    Returns:
        The estimated cost in USD.
    """
    # Simple fuzzy matching for model names
    sorted_keys = sorted(PRICING.keys(), key=len, reverse=True)

    pricing = None
    for key in sorted_keys:
        if key in model:
            pricing = PRICING[key]
            break

    if not pricing:
        logger.warning(f"No pricing found for model {model}. Cost will be 0.0.")
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class OpenAIClientAsync:
    """Async implementation of LLMClient using OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize the OpenAIClientAsync.

        Args:
            api_key: Optional API key. If not provided, reads from OPENAI_API_KEY env var.
            client: Optional pre-configured AsyncOpenAI client instance.
            http_client: Optional httpx.AsyncClient for connection pooling.
        """
        self._internal_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient()

        if client:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), http_client=self._http_client)

    async def __aenter__(self) -> "OpenAIClientAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._http_client.aclose()
        # Close the OpenAI client session if needed, but OpenAI client doesn't strictly require
        # explicit close if we manage the httpx client. However, it's good practice.
        await self.client.close()

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the OpenAI LLM asynchronously.

        Args:
            messages: A list of message dictionaries (role, content).
            model: The model identifier to use (default: 'gpt-4o').
            temperature: Sampling temperature (default: 0.0).
            **kwargs: Additional arguments passed to the OpenAI API.

        Returns:
            LLMResponse containing the content and usage statistics.

        Raises:
            ValueError: If streaming is requested (not supported).
        """
        model = model or "gpt-4o"

        if kwargs.get("stream"):
            raise ValueError("Streaming is not supported by OpenAIClientAsync.")

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            usage = response.usage
            if usage:
                cost = calculate_openai_cost(model, usage.prompt_tokens, usage.completion_tokens)
                usage_stats = UsageStats(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cost_usd=cost,
                )
            else:
                usage_stats = UsageStats()

            return LLMResponse(content=content, usage=usage_stats)

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise


class OpenAIClient:
    """Sync Facade for OpenAIClientAsync."""

    def __init__(
        self,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize the OpenAIClient Facade.

        Args:
            api_key: Optional API key.
            client: Optional AsyncOpenAI client.
            http_client: Optional httpx.AsyncClient.
        """
        self._async = OpenAIClientAsync(api_key=api_key, client=client, http_client=http_client)

    def __enter__(self) -> "OpenAIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            anyio.run(self._async.__aexit__, *args)
        except Exception:
            # anyio.run might propagate ExceptionGroup if not handled.
            # actually anyio.run re-raises the exception if one occurred in the coroutine.
            raise

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the OpenAI LLM (Synchronous Facade).
        """
        response: LLMResponse = anyio.run(
            lambda: self._async.generate(
                messages=messages,
                model=model,
                temperature=temperature,
                **kwargs,
            )
        )
        return response


class BudgetAwareLLMClientAsync:
    """Async Wrapper for AsyncLLMClient that enforces a budget."""

    def __init__(self, client: AsyncLLMClient, budget_manager: BudgetManager):
        """
        Initialize the BudgetAwareLLMClientAsync.

        Args:
            client: The underlying AsyncLLMClient to wrap.
            budget_manager: The BudgetManager to track usage.
        """
        self.client = client
        self.budget_manager = budget_manager

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate response and consume budget.

        Checks budget before and updates budget after the call.

        Args:
            messages: A list of message dictionaries.
            model: The model identifier.
            temperature: Sampling temperature.
            **kwargs: Additional arguments.

        Returns:
            LLMResponse.
        """
        # 0. Check Budget Pre-flight
        self.budget_manager.check_budget()

        # 1. Generate
        response = await self.client.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs,
        )

        # 2. Track Budget
        self.budget_manager.consume(response.usage)

        return response


class BudgetAwareLLMClient:
    """Wrapper for LLMClient (Sync) that enforces a budget."""

    def __init__(self, client: LLMClient, budget_manager: BudgetManager):
        """
        Initialize the BudgetAwareLLMClient.

        Args:
            client: The underlying LLMClient to wrap.
            budget_manager: The BudgetManager to track usage.
        """
        self.client = client
        self.budget_manager = budget_manager

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate response and consume budget.
        """
        # 0. Check Budget Pre-flight
        self.budget_manager.check_budget()

        # 1. Generate
        response = self.client.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs,
        )

        # 2. Track Budget
        self.budget_manager.consume(response.usage)

        return response


class OpenAIEmbeddingClientAsync:
    """Async Implementation of EmbeddingProvider using OpenAI."""

    def __init__(
        self,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize the OpenAIEmbeddingClientAsync.

        Args:
            api_key: Optional API key. If not provided, reads from OPENAI_API_KEY env var.
            client: Optional pre-configured AsyncOpenAI client instance.
            http_client: Optional httpx.AsyncClient for connection pooling.
        """
        self._internal_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient()

        if client:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), http_client=self._http_client)

    async def __aenter__(self) -> "OpenAIEmbeddingClientAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._http_client.aclose()
        await self.client.close()

    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts (with batching) asynchronously.

        Args:
            texts: List of strings to embed.
            model: The embedding model to use (default: 'text-embedding-3-small').

        Returns:
            EmbeddingResponse containing embeddings and usage stats.
        """
        model = model or "text-embedding-3-small"
        batch_size = 500
        all_embeddings = []
        total_prompt_tokens = 0
        total_cost = 0.0

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self.client.embeddings.create(input=batch, model=model)

                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)

                if response.usage:
                    tokens = response.usage.prompt_tokens
                    total_prompt_tokens += tokens
                    total_cost += calculate_openai_cost(model, tokens, 0)

            return EmbeddingResponse(
                embeddings=all_embeddings,
                usage=UsageStats(
                    prompt_tokens=total_prompt_tokens,
                    total_tokens=total_prompt_tokens,
                    cost_usd=total_cost,
                ),
            )

        except Exception as e:
            logger.error(f"OpenAI Embedding failed: {e}")
            raise


class OpenAIEmbeddingClient:
    """Sync Facade for OpenAIEmbeddingClientAsync."""

    def __init__(
        self,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self._async = OpenAIEmbeddingClientAsync(api_key=api_key, client=client, http_client=http_client)

    def __enter__(self) -> "OpenAIEmbeddingClient":
        return self

    def __exit__(self, *args: Any) -> None:
        anyio.run(self._async.__aexit__, *args)

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        response: EmbeddingResponse = anyio.run(lambda: self._async.embed(texts=texts, model=model))
        return response


class BudgetAwareEmbeddingProviderAsync:
    """Async Wrapper for AsyncEmbeddingProvider that enforces a budget."""

    def __init__(self, provider: AsyncEmbeddingProvider, budget_manager: BudgetManager):
        """
        Initialize the BudgetAwareEmbeddingProviderAsync.

        Args:
            provider: The underlying AsyncEmbeddingProvider to wrap.
            budget_manager: The BudgetManager to track usage.
        """
        self.provider = provider
        self.budget_manager = budget_manager

    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """
        Generate embeddings and consume budget.

        Args:
            texts: List of strings to embed.
            model: The embedding model to use.

        Returns:
            EmbeddingResponse.
        """
        self.budget_manager.check_budget()
        response = await self.provider.embed(texts, model)
        self.budget_manager.consume(response.usage)
        return response


class BudgetAwareEmbeddingProvider:
    """Wrapper for EmbeddingProvider (Sync) that enforces a budget."""

    def __init__(self, provider: EmbeddingProvider, budget_manager: BudgetManager):
        """
        Initialize the BudgetAwareEmbeddingProvider.

        Args:
            provider: The underlying EmbeddingProvider to wrap.
            budget_manager: The BudgetManager to track usage.
        """
        self.provider = provider
        self.budget_manager = budget_manager

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """
        Generate embeddings and consume budget.
        """
        self.budget_manager.check_budget()
        response = self.provider.embed(texts, model)
        self.budget_manager.consume(response.usage)
        return response
