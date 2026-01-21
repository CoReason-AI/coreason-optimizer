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

from openai import OpenAI

from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.core.interfaces import (
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
    """Calculate the cost of an OpenAI API call."""
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


class OpenAIClient:
    """Concrete implementation of LLMClient using OpenAI."""

    def __init__(self, api_key: str | None = None, client: OpenAI | None = None):
        if client:
            self.client = client
        else:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        model = model or "gpt-4o"

        if kwargs.get("stream"):
            raise ValueError("Streaming is not supported by OpenAIClient.")

        try:
            response = self.client.chat.completions.create(
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


class BudgetAwareLLMClient:
    """Wrapper for LLMClient that enforces a budget."""

    def __init__(self, client: LLMClient, budget_manager: BudgetManager):
        self.client = client
        self.budget_manager = budget_manager

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response and consume budget."""
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


class OpenAIEmbeddingClient:
    """Implementation of EmbeddingProvider using OpenAI."""

    def __init__(self, api_key: str | None = None, client: OpenAI | None = None):
        if client:
            self.client = client
        else:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """Generate embeddings for a list of texts (with batching)."""
        model = model or "text-embedding-3-small"
        batch_size = 500
        all_embeddings = []
        total_prompt_tokens = 0
        total_cost = 0.0

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self.client.embeddings.create(input=batch, model=model)

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


class BudgetAwareEmbeddingProvider:
    """Wrapper for EmbeddingProvider that enforces a budget."""

    def __init__(self, provider: EmbeddingProvider, budget_manager: BudgetManager):
        self.provider = provider
        self.budget_manager = budget_manager

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        self.budget_manager.check_budget()
        response = self.provider.embed(texts, model)
        self.budget_manager.consume(response.usage)
        return response
