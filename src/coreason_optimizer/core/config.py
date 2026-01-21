# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

<<<<<<< HEAD
=======
"""
Configuration models for the Optimizer.

This module defines the configuration settings for the optimization process,
including model selection, metrics, and budget limits.
"""

from typing import Literal

>>>>>>> c9d6380 (feat: enhance documentation, code standards, and dependencies (#31) (#32))
from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """
    Configuration for the Prompt Optimizer.

    Attributes:
        target_model: The identifier of the target LLM to optimize for.
        meta_model: The identifier of the meta-LLM used for instruction optimization.
        metric: The metric function identifier to use for evaluation.
        selector_type: The strategy to use for selecting few-shot examples.
        embedding_model: The identifier of the embedding model (used if selector_type is semantic).
        max_bootstrapped_demos: Maximum number of few-shot examples to bootstrap.
        max_rounds: Maximum number of optimization rounds.
        budget_limit_usd: Maximum budget in USD for the optimization run.
    """

    target_model: str = Field(
        default="gpt-4o",
        description="The identifier of the target LLM to optimize for.",
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
