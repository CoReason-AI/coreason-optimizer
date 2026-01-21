# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from coreason_optimizer.core.models import TrainingExample


class UsageStats(BaseModel):
    """Token usage statistics for an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class LLMResponse(BaseModel):
    """Standardized response from an LLM."""

    content: str
    usage: UsageStats


class EmbeddingResponse(BaseModel):
    """Standardized response from an embedding provider."""

    embeddings: list[list[float]]
    usage: UsageStats


@runtime_checkable
class Construct(Protocol):
    """Protocol representing a coreason-construct Agent."""

    @property
    def inputs(self) -> list[str]: ...  # pragma: no cover

    @property
    def outputs(self) -> list[str]: ...  # pragma: no cover

    @property
    def system_prompt(self) -> str: ...  # pragma: no cover


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for a generic LLM client."""

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        ...  # pragma: no cover


@runtime_checkable
class Metric(Protocol):
    """Protocol for a scoring function."""

    def __call__(self, prediction: str, reference: Any, **kwargs: Any) -> float:
        """Calculate a score for the prediction against the reference."""
        ...  # pragma: no cover


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for an embedding provider."""

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        ...  # pragma: no cover


class PromptOptimizer(ABC):
    """Abstract base class for prompt optimization strategies."""

    @abstractmethod
    def compile(
        self,
        agent: Construct,
        trainset: list[TrainingExample],
        valset: list[TrainingExample],
    ) -> Any:
        """Run the optimization loop to produce an optimized manifest."""
        pass  # pragma: no cover
