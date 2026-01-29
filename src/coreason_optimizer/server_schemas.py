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
API Schemas for the Optimization Microservice.
"""

from typing import Literal

from pydantic import BaseModel, Field

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.models import TrainingExample


class AgentDefinition(BaseModel):
    """
    Schema for an agent definition in the Optimization request.

    Mirrors the Construct protocol properties.
    """

    system_prompt: str = Field(..., description="The initial system prompt text.")
    inputs: list[str] = Field(..., description="List of input field names.")
    outputs: list[str] = Field(..., description="List of output field names.")


class OptimizationRequest(BaseModel):
    """
    Request schema for the optimization endpoint.
    """

    agent: AgentDefinition = Field(..., description="The agent to optimize.")
    dataset: list[TrainingExample] = Field(..., description="List of training examples.")
    config: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimization configuration.")
    strategy: Literal["mipro", "bootstrap"] = Field(default="mipro", description="Optimization strategy to use.")
