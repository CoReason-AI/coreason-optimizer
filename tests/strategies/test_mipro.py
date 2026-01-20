# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mipro import MiproOptimizer


class MockConstruct:
    system_prompt = "Original Instruction"
    inputs = ["input"]
    outputs = ["output"]


@pytest.fixture  # type: ignore[misc]
def mock_llm() -> MagicMock:
    llm = MagicMock()
    # Default response
    llm.generate.return_value = LLMResponse(content="default response", usage=UsageStats())
    return llm


@pytest.fixture  # type: ignore[misc]
def mock_metric() -> MagicMock:
    # Simple metric: returns 1.0 if prediction == "correct", else 0.0
    def metric_fn(prediction: str, reference: Any, **kwargs: Any) -> float:
        return 1.0 if prediction == "correct" else 0.0

    return MagicMock(side_effect=metric_fn)


def test_mipro_optimizer_flow(mock_llm: MagicMock, mock_metric: MagicMock) -> None:
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

    # 2. Setup LLM Behavior

    def side_effect(messages: list[dict[str, str]], model: str | None = None, **kwargs: Any) -> LLMResponse:
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

    mock_llm.generate.side_effect = side_effect

    # 3. Run MIPRO
    # Reduce candidates for speed in tests
    optimizer = MiproOptimizer(mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1)

    manifest = optimizer.compile(agent, trainset, valset)

    # 4. Assertions
    assert manifest.optimized_instruction == "Mutated Instruction"
    assert manifest.performance_metric == 1.0  # Should score 1.0 on valset if best instruction is used

    # Verify calls
    meta_calls = [call for call in mock_llm.generate.mock_calls if call.kwargs.get("model") == "meta-model"]
    assert len(meta_calls) >= 1


def test_mipro_optimizer_no_failures(mock_llm: MagicMock, mock_metric: MagicMock) -> None:
    """Test MIPRO when baseline is perfect (no failures)."""
    config = OptimizerConfig()

    trainset = [TrainingExample(inputs={"q": "1"}, reference="A")]
    agent = MockConstruct()

    # LLM always correct
    mock_llm.generate.return_value = LLMResponse(content="correct", usage=UsageStats())

    optimizer = MiproOptimizer(mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1)

    manifest = optimizer.compile(agent, trainset, [])

    # Should keep original instruction
    assert manifest.optimized_instruction == agent.system_prompt
    assert manifest.performance_metric == 1.0


def test_mipro_optimizer_resilience(mock_llm: MagicMock, mock_metric: MagicMock) -> None:
    """Test that MIPRO continues despite errors in LLM calls and Mutator."""
    config = OptimizerConfig(target_model="tgt", meta_model="meta")
    trainset = [TrainingExample(inputs={"q": "1"}, reference="A")]
    agent = MockConstruct()

    # Case 1: LLM raises on generate (Diagnosis & Evaluation)
    mock_llm.generate.side_effect = Exception("LLM Error")

    # Case 2: Mutator raises on mutate
    # We use patch to mock the Mutator class used inside MiproOptimizer
    with patch("coreason_optimizer.strategies.mipro.LLMInstructionMutator") as MockMutatorClass:
        mock_mutator_instance = MockMutatorClass.return_value
        mock_mutator_instance.mutate.side_effect = Exception("Mutator Error")

        optimizer = MiproOptimizer(
            mock_llm, mock_metric, config, num_instruction_candidates=1, num_fewshot_combinations=1
        )

        manifest = optimizer.compile(agent, trainset, [])

        # Should survive all errors
        assert manifest.optimized_instruction == agent.system_prompt
        assert manifest.few_shot_examples == []
        # Score 0.0 because default _evaluate_candidate returns 0.0 on error
        assert manifest.performance_metric == 0.0
