# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """Configuration for the Prompt Optimizer."""

    target_model: str = Field(
        default="gpt-4o",
        description="The identifier of the target LLM to optimize for.",
    )
    meta_model: str = Field(
        default="gpt-4o",
        description="The identifier of the meta-LLM used for instruction optimization.",
    )
    metric: str = Field(
        default="exact_match",
        description="The metric function identifier to use for evaluation.",
    )
    max_bootstrapped_demos: int = Field(
        default=4,
        ge=0,
        description="Maximum number of few-shot examples to bootstrap.",
    )
    max_rounds: int = Field(
        default=10,
        gt=0,
        description="Maximum number of optimization rounds.",
    )
    budget_limit_usd: float = Field(
        default=10.0,
        gt=0.0,
        description="Maximum budget in USD for the optimization run.",
    )
