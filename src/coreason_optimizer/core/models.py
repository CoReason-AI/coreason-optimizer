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
Pydantic data models used across the library.

This module defines the core data structures for training examples and
the output manifest of the optimization process.
"""

from typing import Any

from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """
    A single example for training or few-shot prompting.

    Attributes:
        inputs: Input variables mapping to the agent's expected inputs.
        reference: The ground truth or expected output for the example.
        metadata: Optional metadata (e.g. source, tags).
    """

    inputs: dict[str, Any] = Field(..., description="Input variables mapping to the agent's expected inputs.")
    reference: Any = Field(..., description="The ground truth or expected output for the example.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata (e.g. source, tags).")


class OptimizedManifest(BaseModel):
    """
    The output artifact of the optimization process.

    Attributes:
        agent_id: The unique identifier of the agent.
        base_model: The base LLM model used.
        optimized_instruction: The optimized system prompt.
        few_shot_examples: Selected few-shot examples.
        performance_metric: The score achieved on the validation set.
        optimization_run_id: Unique ID for this optimization run.
    """

    agent_id: str = Field(..., description="The unique identifier of the agent.")
    base_model: str = Field(..., description="The base LLM model used.")
    optimized_instruction: str = Field(..., description="The optimized system prompt.")
    few_shot_examples: list[TrainingExample] = Field(default_factory=list, description="Selected few-shot examples.")
    performance_metric: float = Field(..., description="The score achieved on the validation set.")
    optimization_run_id: str = Field(..., description="Unique ID for this optimization run.")
