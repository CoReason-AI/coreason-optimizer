# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_optimizer.core.budget import BudgetExceededError
from coreason_optimizer.core.client import OpenAIClient, OpenAIEmbeddingClient
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import (
    EmbeddingResponse,
    LLMResponse,
    UsageStats,
)
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mipro import MiproOptimizer


class MockConstruct:
    system_prompt = "Original Instruction"
    inputs = ["input"]
    outputs = ["output"]


@pytest.fixture
def mock_llm() -> OpenAIClient:
    # Need to mock OpenAIClient (Sync Facade) structure
    # The Facade initialization checks if it's an instance of OpenAIClient
    # So we need to ensure isinstance(llm, OpenAIClient) passes?
    # MiproOptimizer init:
    # if isinstance(llm_client, OpenAIClient): ...

    # We can just instantiate a real OpenAIClient with mocked internals
    # But init requires api_key if no client provided, but we pass nothing so internal async needs key

    with patch("coreason_optimizer.core.client.OpenAIClientAsync"):
        # Mock the async client creation
        client = OpenAIClient(api_key="test")

    # Mock internal async client
    client._async_client = AsyncMock()
    # Mock context manager
    client._async_client.__aenter__.return_value = client._async_client

    # Default response for async generate
    async_response = LLMResponse(content="default response", usage=UsageStats())
    client._async_client.generate.return_value = async_response

    return client


@pytest.fixture
def mock_metric() -> MagicMock:
    # Simple metric: returns 1.0 if prediction == "correct", else 0.0
    def metric_fn(prediction: str, reference: Any, **kwargs: Any) -> float:
        return 1.0 if prediction == "correct" else 0.0

    return MagicMock(side_effect=metric_fn)


def test_mipro_optimizer_flow(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test the complete MIPRO flow."""
    config = OptimizerConfig(target_model="test-model", meta_model="meta-model", max_bootstrapped_demos=2)

    # 1. Setup Data
    trainset = [
        TrainingExample(inputs={"q": "1"}, reference="A"),
        TrainingExample(inputs={"q": "2"}, reference="B"),
    ]
    valset = [
        TrainingExample(inputs={"q": "3"}, reference="C"),
    ]
    agent = MockConstruct()

    # 2. Setup LLM Behavior (Async)

    async def side_effect(messages: list[dict[str, str]], model: str | None = None, **kwargs: Any) -> LLMResponse:
        content = messages[0]["content"]

        # Meta-LLM Mutation Call
        if model == "meta-model":
            return LLMResponse(content="Mutated Instruction", usage=UsageStats())

        # Target Model Calls
        # Diagnosis: Original Instruction -> Let's make it fail
        if "Original Instruction" in content:
            return LLMResponse(content="wrong", usage=UsageStats())

        # Grid Search: Mutated Instruction -> Let's make it succeed
        if "Mutated Instruction" in content:
            return LLMResponse(content="correct", usage=UsageStats())

        return LLMResponse(content="unknown", usage=UsageStats())

    # Cast to Any to satisfy mypy for mocked attributes
    cast(Any, mock_llm._async_client.generate).side_effect = side_effect

    # 3. Run MIPRO
    # Reduce candidates for speed in tests
    optimizer = MiproOptimizer(mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1)

    manifest = optimizer.compile(agent, trainset, valset)

    # 4. Assertions
    assert manifest.optimized_instruction == "Mutated Instruction"
    assert manifest.performance_metric == 1.0  # Should score 1.0 on valset if best instruction is used

    # Verify calls
    # Cast to Any to bypass mypy check on mock_calls
    mock_calls = cast(Any, mock_llm._async_client.generate).mock_calls
    meta_calls = [call for call in mock_calls if call.kwargs.get("model") == "meta-model"]
    assert len(meta_calls) >= 1


def test_mipro_optimizer_no_failures(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test MIPRO when baseline is perfect (no failures)."""
    config = OptimizerConfig()

    trainset = [TrainingExample(inputs={"q": "1"}, reference="A")]
    agent = MockConstruct()

    # LLM always correct
    cast(Any, mock_llm._async_client.generate).return_value = LLMResponse(content="correct", usage=UsageStats())
    cast(Any, mock_llm._async_client.generate).side_effect = None  # Clear side effect from fixture

    optimizer = MiproOptimizer(mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1)

    manifest = optimizer.compile(agent, trainset, [])

    # Should keep original instruction
    assert manifest.optimized_instruction == agent.system_prompt
    assert manifest.performance_metric == 1.0


def test_mipro_optimizer_resilience(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test that MIPRO continues despite errors in LLM calls and Mutator."""
    config = OptimizerConfig(target_model="tgt", meta_model="meta")
    trainset = [TrainingExample(inputs={"q": "1"}, reference="A")]
    agent = MockConstruct()

    # Case 1: LLM raises on generate (Diagnosis & Evaluation)
    cast(Any, mock_llm._async_client.generate).side_effect = Exception("LLM Error")

    # Case 2: Mutator raises on mutate
    # We use patch to mock the Mutator class used inside MiproOptimizerAsync
    # MiproOptimizer uses MiproOptimizerAsync which uses LLMInstructionMutatorAsync
    with patch("coreason_optimizer.strategies.mipro.LLMInstructionMutatorAsync") as MockMutatorClass:
        mock_mutator_instance = MockMutatorClass.return_value
        # mutate is async
        mock_mutator_instance.mutate = AsyncMock(side_effect=Exception("Mutator Error"))

        optimizer = MiproOptimizer(
            mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1
        )

        manifest = optimizer.compile(agent, trainset, [])

        # Should survive all errors
        assert manifest.optimized_instruction == agent.system_prompt
        assert manifest.few_shot_examples == []
        # Score 0.0 because default _evaluate_candidate returns 0.0 on error
        assert manifest.performance_metric == 0.0


def test_mipro_empty_trainset(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test behavior with empty training set."""
    config = OptimizerConfig()
    agent = MockConstruct()

    optimizer = MiproOptimizer(mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1)

    # Empty trainset
    manifest = optimizer.compile(agent, [], [])

    assert manifest.optimized_instruction == agent.system_prompt
    assert manifest.few_shot_examples == []
    assert manifest.performance_metric == 0.0


def test_mipro_complex_scoring(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test that the optimizer selects the best combination from multiple candidates."""
    config = OptimizerConfig(target_model="tgt", meta_model="meta-model")
    agent = MockConstruct()

    # Setup data
    trainset = [TrainingExample(inputs={"q": "1"}, reference="A")]
    ex1 = TrainingExample(inputs={"q": "ex"}, reference="ref")

    # Metric logic: parses score from response
    def score_parser(prediction: str, reference: Any, **kwargs: Any) -> float:
        try:
            return float(prediction)
        except ValueError:
            return 0.0

    mock_metric.side_effect = score_parser

    # LLM logic
    async def complex_side_effect(
        messages: list[dict[str, str]], model: str | None = None, **kwargs: Any
    ) -> LLMResponse:
        content = messages[0]["content"]

        # Mutation
        if model == "meta-model":
            return LLMResponse(content="Better Instruction", usage=UsageStats())

        # Diagnosis (Original, []) -> Fail to trigger mutation
        # Evaluated on trainset (q=1)
        if "Original Instruction" in content and "Better Instruction" not in content and "### Examples" not in content:
            return LLMResponse(content="0.0", usage=UsageStats())

        # Evaluation (Grid Search)
        # Check combination
        has_better = "Better Instruction" in content
        has_examples = "Input: q: ex" in content

        if has_better and has_examples:
            return LLMResponse(content="0.95", usage=UsageStats())
        elif has_better:
            return LLMResponse(content="0.8", usage=UsageStats())
        elif has_examples:
            return LLMResponse(content="0.6", usage=UsageStats())
        else:  # Original, no examples
            return LLMResponse(content="0.5", usage=UsageStats())

    cast(Any, mock_llm._async_client.generate).side_effect = complex_side_effect

    # Patch Selector to always return [ex1]
    # MiproOptimizerAsync uses RandomSelectorAsync
    with patch("coreason_optimizer.strategies.mipro.RandomSelectorAsync") as MockSelectorClass:
        mock_selector = MockSelectorClass.return_value
        # select is async
        mock_selector.select = AsyncMock(return_value=[ex1])

        optimizer = MiproOptimizer(
            mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1
        )

        manifest = optimizer.compile(agent, trainset, [])

        assert manifest.optimized_instruction == "Better Instruction"
        assert manifest.few_shot_examples == [ex1]
        assert manifest.performance_metric == 0.95


def test_mipro_semantic_selector_integration(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test that MIPRO uses SemanticSelector when configured."""
    config = OptimizerConfig(selector_type="semantic", embedding_model="emb-model")

    # Need OpenAIEmbeddingClient facade
    with patch("coreason_optimizer.core.client.OpenAIEmbeddingClientAsync"):
        mock_embedder = OpenAIEmbeddingClient(api_key="test")

    mock_embedder._async_client = AsyncMock()
    # Mock return value
    mock_embedder._async_client.embed.return_value = EmbeddingResponse(embeddings=[[0.1, 0.2]], usage=UsageStats())

    optimizer = MiproOptimizer(
        mock_llm,
        mock_metric,
        config,
        embedding_provider=mock_embedder,
        num_instruction_candidates=1,
        num_fewshot_combinations=1,
    )

    # Verify that the selector is SemanticSelectorAsync (inside async optimizer)
    # The Facade MiproOptimizer has ._async which has .selector
    # Note: .selector is on the async instance, which is private _async
    # We access it for testing purposes
    # Cast to Any to avoid mypy error about attribute access on private member
    assert isinstance(
        cast(Any, optimizer._async.selector).embedding_provider.provider, AsyncMock
    )  # It's wrapped in BudgetAware
    # Actually checking type might be hard due to patching or imports
    # But checking if we passed the provider correctly

    # Run a simple compile to ensure flow works (no crash)
    agent = MockConstruct()
    trainset = [TrainingExample(inputs={"q": "1"}, reference="A")]

    manifest = optimizer.compile(agent, trainset, [])
    assert manifest.base_model == config.target_model


def test_mipro_missing_embedding_provider(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test that MIPRO raises ValueError if semantic selector is requested but no provider is given."""
    config = OptimizerConfig(selector_type="semantic")

    with pytest.raises(ValueError, match="Embedding provider is required"):
        MiproOptimizer(mock_llm, mock_metric, config)


def test_mipro_semantic_budget_exceeded(mock_llm: OpenAIClient, mock_metric: MagicMock) -> None:
    """Test that BudgetExceededError propagates during semantic selection."""
    # Set a very low budget
    config = OptimizerConfig(
        selector_type="semantic",
        embedding_model="expensive-model",
        budget_limit_usd=0.0000001,
        max_bootstrapped_demos=1,
    )

    # Mock Embedder that consumes budget
    with patch("coreason_optimizer.core.client.OpenAIEmbeddingClientAsync"):
        mock_embedder = OpenAIEmbeddingClient(api_key="test")
    mock_embedder._async_client = AsyncMock()

    # The UsageStats cost should exceed budget
    mock_embedder._async_client.embed.return_value = EmbeddingResponse(
        embeddings=[[0.1], [0.1]], usage=UsageStats(cost_usd=1.0)
    )

    optimizer = MiproOptimizer(
        mock_llm,
        mock_metric,
        config,
        embedding_provider=mock_embedder,
        num_instruction_candidates=1,
        num_fewshot_combinations=1,
    )

    agent = MockConstruct()
    trainset = [
        TrainingExample(inputs={"q": "1"}, reference="A"),
        TrainingExample(inputs={"q": "2"}, reference="B"),
    ]

    # Should raise BudgetExceededError when selecting examples
    with pytest.raises(BudgetExceededError):
        optimizer.compile(agent, trainset, [])
