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
Async LLM Client implementations for interacting with OpenAI API.
"""

import os
from types import TracebackType
from typing import Any, Optional

from openai import AsyncOpenAI

from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.core.interfaces import (
    EmbeddingProviderAsync,
    EmbeddingResponse,
    LLMClientAsync,
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

    def __init__(self, api_key: Optional[str] = None, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the OpenAIClientAsync.

        Args:
            api_key: Optional API key. If not provided, reads from OPENAI_API_KEY env var.
            client: Optional pre-configured AsyncOpenAI client instance.
        """
        self._internal_client = client is None
        self.client = client or AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def __aenter__(self) -> "OpenAIClientAsync":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._internal_client:  # pragma: no cover
            await self.client.close()  # pragma: no cover

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the OpenAI LLM asynchronously.
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
            logger.error(f"OpenAI Async API call failed: {e}")
            raise


class OpenAIEmbeddingClientAsync:
    """Async implementation of EmbeddingProvider using OpenAI."""

    def __init__(self, api_key: Optional[str] = None, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the OpenAIEmbeddingClientAsync.

        Args:
            api_key: Optional API key. If not provided, reads from OPENAI_API_KEY env var.
            client: Optional pre-configured AsyncOpenAI client instance.
        """
        self._internal_client = client is None
        self.client = client or AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def __aenter__(self) -> "OpenAIEmbeddingClientAsync":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._internal_client:
            await self.client.close()

    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts asynchronously (with batching).
        """
        model = model or "text-embedding-3-small"
        batch_size = 500
        all_embeddings = []
        total_prompt_tokens = 0
        total_cost = 0.0

        try:
            # anyio.to_thread.run_sync not needed for I/O bound tasks like this as we use async client.
            # But we can process batches concurrently if needed.
            # For simplicity, we process batches sequentially but asynchronously.

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
            logger.error(f"OpenAI Async Embedding failed: {e}")
            raise


class BudgetAwareLLMClientAsync:
    """Wrapper for LLMClientAsync that enforces a budget."""

    def __init__(self, client: LLMClientAsync, budget_manager: BudgetManager):
        self.client = client
        self.budget_manager = budget_manager

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        self.budget_manager.check_budget()
        response = await self.client.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs,
        )
        self.budget_manager.consume(response.usage)
        return response


class BudgetAwareEmbeddingProviderAsync:
    """Wrapper for EmbeddingProviderAsync that enforces a budget."""

    def __init__(self, provider: EmbeddingProviderAsync, budget_manager: BudgetManager):
        self.provider = provider
        self.budget_manager = budget_manager

    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        self.budget_manager.check_budget()
        response = await self.provider.embed(texts, model)
        self.budget_manager.consume(response.usage)
        return response
