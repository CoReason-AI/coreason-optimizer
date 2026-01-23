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
Core interfaces and protocols.

This module defines the abstract base classes, protocols, and shared data models
used throughout the library, ensuring loose coupling and type safety.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from coreason_optimizer.core.models import TrainingExample


class UsageStats(BaseModel):
    """
    Token usage statistics for an LLM call.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
        cost_usd: Estimated cost in USD.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class LLMResponse(BaseModel):
    """
    Standardized response from an LLM.

    Attributes:
        content: The text content of the response.
        usage: Usage statistics for the call.
    """

    content: str
    usage: UsageStats


class EmbeddingResponse(BaseModel):
    """
    Standardized response from an embedding provider.

    Attributes:
        embeddings: List of embedding vectors.
        usage: Usage statistics for the call.
    """

    embeddings: list[list[float]]
    usage: UsageStats


@runtime_checkable
class Construct(Protocol):
    """
    Protocol representing a coreason-construct Agent.

    Attributes:
        inputs: List of input field names.
        outputs: List of output field names.
        system_prompt: The initial system prompt text.
    """

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
        """
        Generate a response from the LLM.

        Args:
            messages: A list of message dictionaries (role, content).
            model: The model identifier to use.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The LLM response containing content and usage stats.
        """
        ...  # pragma: no cover


@runtime_checkable
class AsyncLLMClient(Protocol):
    """Protocol for a generic Async LLM client."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM asynchronously.

        Args:
            messages: A list of message dictionaries (role, content).
            model: The model identifier to use.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The LLM response containing content and usage stats.
        """
        ...  # pragma: no cover


@runtime_checkable
class Metric(Protocol):
    """Protocol for a scoring function."""

    def __call__(self, prediction: str, reference: Any, **kwargs: Any) -> float:
        """
        Calculate a score for the prediction against the reference.

        Args:
            prediction: The model's output.
            reference: The ground truth value.
            **kwargs: Additional arguments for the metric function.

        Returns:
            A float score (typically 0.0 to 1.0).
        """
        ...  # pragma: no cover


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for an embedding provider."""

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            model: The embedding model to use.

        Returns:
            An EmbeddingResponse containing vectors and usage stats.
        """
        ...  # pragma: no cover


@runtime_checkable
class AsyncEmbeddingProvider(Protocol):
    """Protocol for an async embedding provider."""

    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts asynchronously.

        Args:
            texts: List of strings to embed.
            model: The embedding model to use.

        Returns:
            An EmbeddingResponse containing vectors and usage stats.
        """
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
        """
        Run the optimization loop to produce an optimized manifest.

        Args:
            agent: The draft agent to optimize.
            trainset: List of examples for training/bootstrapping.
            valset: List of examples for validation/evaluation.

        Returns:
            An optimized manifest object (specific type depends on implementation).
        """
        pass  # pragma: no cover
