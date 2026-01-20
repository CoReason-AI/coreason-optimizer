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

from coreason_optimizer.core.interfaces import LLMResponse, UsageStats
from coreason_optimizer.utils.logger import logger

# Pricing per 1M tokens (approximate as of early 2025)
PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
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
