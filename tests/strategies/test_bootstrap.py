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

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMResponse, UsageStats
from coreason_optimizer.core.metrics import ExactMatch
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.bootstrap import BootstrapFewShot


class MockAgent:
    @property
    def inputs(self) -> list[str]:
        return ["question"]

    @property
    def outputs(self) -> list[str]:
        return ["answer"]

    @property
    def system_prompt(self) -> str:
        return "Answer the question."


class MockLLMClient:
    def __init__(self):
        self.calls: list[Any] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(messages)
        prompt = messages[0]["content"]

        # Parse prompt to find the active user input
        # Structure is:
        # ### System Instruction
        # ...
        # ### Examples
        # ...
        # ### User Input
        # Input: ...

        parts = prompt.split("### User Input")
        user_input_part = parts[-1] if len(parts) > 1 else prompt

        # Simulate correct answer for "2+2"
        if "Input: question: 2+2" in user_input_part:
            return LLMResponse(content="4", usage=UsageStats())
        # Simulate incorrect answer for "3+3"
        if "Input: question: 3+3" in user_input_part:
            return LLMResponse(content="99", usage=UsageStats())
        # Simulate correct answer for val set "5+5"
        if "Input: question: 5+5" in user_input_part:
            return LLMResponse(content="10", usage=UsageStats())

        return LLMResponse(content="unknown", usage=UsageStats())


class FailingLLMClient:
    def __init__(self, fail_on_train: bool = True):
        self.fail_on_train = fail_on_train

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        prompt = messages[0]["content"]

        # Check if validation (has Examples) or training (no Examples)
        # Note: We can't rely just on "### Examples" string because empty list means no header.
        # But we can rely on our knowledge of the test cases.

        if self.fail_on_train:
            # Check if this is the training call (no 5+5)
            if "5+5" not in prompt:
                raise RuntimeError("LLM Failure")

        # For validation test case below
        if not self.fail_on_train:
            # Train step: succeed to generate a demo
            if "Input: q: 1" in prompt and "### Examples" not in prompt:
                return LLMResponse(content="1", usage=UsageStats())

            # Validation step: fail
            # Validation prompt will have the example we just generated
            if "Input: q: 2" in prompt:
                raise RuntimeError("Validation Failure")

        return LLMResponse(content="42", usage=UsageStats())


def test_bootstrap_few_shot_compile():
    llm = MockLLMClient()
    metric = ExactMatch()
    config = OptimizerConfig(target_model="test-model", max_bootstrapped_demos=1)
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = MockAgent()
    trainset = [
        TrainingExample(inputs={"question": "2+2"}, reference="4"),  # Should pass
        TrainingExample(inputs={"question": "3+3"}, reference="6"),  # Should fail
    ]
    valset = [
        TrainingExample(inputs={"question": "5+5"}, reference="10"),
    ]

    manifest = optimizer.compile(agent, trainset, valset)

    # 1. Verify successful traces mined
    assert len(manifest.few_shot_examples) == 1
    assert manifest.few_shot_examples[0].inputs["question"] == "2+2"

    # 2. Verify manifest fields
    assert manifest.base_model == "test-model"
    assert manifest.performance_metric == 1.0  # Val set should pass
    assert manifest.optimized_instruction == "Answer the question."

    # 3. Verify interaction with LLM
    # Expect 2 calls for training (mining) + 1 call for validation
    assert len(llm.calls) == 3


def test_bootstrap_few_shot_empty_trainset():
    llm = MockLLMClient()
    metric = ExactMatch()
    config = OptimizerConfig()
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = MockAgent()
    manifest = optimizer.compile(agent, [], [])

    assert len(manifest.few_shot_examples) == 0
    assert manifest.performance_metric == 0.0


def test_bootstrap_limit_demos():
    """Test that max_bootstrapped_demos is respected."""
    llm = MockLLMClient()
    metric = ExactMatch()
    # Limit to 1 demo
    config = OptimizerConfig(max_bootstrapped_demos=1)
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = MockAgent()
    # Two passing examples
    trainset = [
        TrainingExample(inputs={"question": "2+2"}, reference="4"),
        TrainingExample(inputs={"question": "2+2"}, reference="4"),
    ]

    manifest = optimizer.compile(agent, trainset, [])

    assert len(manifest.few_shot_examples) == 1


def test_bootstrap_llm_exception_mining():
    """Test exception handling during mining."""
    # This client fails on training
    llm = FailingLLMClient(fail_on_train=True)
    metric = ExactMatch()
    config = OptimizerConfig()
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = MockAgent()
    trainset = [TrainingExample(inputs={"q": "1"}, reference="1")]

    # Should not crash, just produce empty manifest
    manifest = optimizer.compile(agent, trainset, [])
    assert len(manifest.few_shot_examples) == 0


def test_bootstrap_llm_exception_validation():
    """Test exception handling during validation."""
    # This client fails on validation
    llm = FailingLLMClient(fail_on_train=False)
    metric = ExactMatch()
    config = OptimizerConfig(max_bootstrapped_demos=1)
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = MockAgent()
    trainset = [TrainingExample(inputs={"q": "1"}, reference="1")]
    valset = [TrainingExample(inputs={"q": "2"}, reference="2")]

    manifest = optimizer.compile(agent, trainset, valset)

    # Should have 1 example, but score 0.0 because validation failed
    assert len(manifest.few_shot_examples) == 1
    assert manifest.performance_metric == 0.0
