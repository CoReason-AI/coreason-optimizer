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


class OpenAIClient:
    """Concrete implementation of LLMClient using OpenAI."""

    def __init__(self, api_key: str | None = None, client: OpenAI | None = None):
        if client:
            self.client = client
        else:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # Simple fuzzy matching for model names
        # Sort keys by length descending to match specific models first (e.g. gpt-4o-mini before gpt-4o)
        sorted_keys = sorted(PRICING.keys(), key=len, reverse=True)

        pricing = None
        for key in sorted_keys:
            if key in model:
                pricing = PRICING[key]
                break

        if not pricing:
            # Fallback to gpt-4o pricing if unknown, or maybe 0?
            # Let's log a warning and use 0.0 or a default.
            logger.warning(f"No pricing found for model {model}. Cost will be 0.0.")
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

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
            # OpenAI expects specific types for messages, but dicts usually work.
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
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

                usage_stats = UsageStats(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
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

    def __init__(
        self,
        api_key: str | None = None,
        client: OpenAI | None = None,
        budget_manager: BudgetManager | None = None,
    ):
        if client:
            self.client = client
        else:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.budget_manager = budget_manager

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        model = model or "text-embedding-3-small"

        try:
            if self.budget_manager:
                self.budget_manager.check_budget()

            # OpenAI embedding call
            # Input can be list of strings
            response = self.client.embeddings.create(input=texts, model=model)

            embeddings = [data.embedding for data in response.data]

            if self.budget_manager and response.usage:
                tokens = response.usage.prompt_tokens
                # Cost calculation
                price_per_m = PRICING.get(model, {}).get("input", 0.02)
                cost = (tokens / 1_000_000) * price_per_m

                self.budget_manager.consume(
                    UsageStats(
                        prompt_tokens=tokens,
                        completion_tokens=0,
                        total_tokens=tokens,
                        cost_usd=cost,
                    )
                )

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI Embedding failed: {e}")
            raise
