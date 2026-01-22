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

import pytest

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMResponse, UsageStats
from coreason_optimizer.core.metrics import ExactMatch
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.bootstrap import BootstrapFewShot


class GenericMockAgent:
    def __init__(self, system_prompt: str = "Default system prompt") -> None:
        self._system_prompt = system_prompt

    @property
    def inputs(self) -> list[str]:
        return ["input"]

    @property
    def outputs(self) -> list[str]:
        return ["output"]

    @property
    def system_prompt(self) -> str:
        return self._system_prompt


class ComplexMockLLMClient:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(messages)
        prompt = messages[0]["content"]

        # Case 1: Non-string inputs
        # Formatter should convert 42 to "42"
        if "Input: count: 42" in prompt:
            return LLMResponse(content="valid", usage=UsageStats())

        # Case 2: List reference
        if "Input: q: color" in prompt:
            return LLMResponse(content="red", usage=UsageStats())

        # Case 3: Multiline prompt
        if "### System Instruction\nLine 1\nLine 2" in prompt:
            # We just need to return something that matches reference to pass
            if "Input: q: multi" in prompt:
                return LLMResponse(content="yes", usage=UsageStats())

        return LLMResponse(content="unknown", usage=UsageStats())


@pytest.mark.asyncio
async def test_bootstrap_non_string_inputs() -> None:
    """Test that integer/float inputs are correctly formatted and processed."""
    llm = ComplexMockLLMClient()
    metric = ExactMatch()
    config = OptimizerConfig(max_bootstrapped_demos=1)
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = GenericMockAgent()
    # Input is an integer
    trainset = [
        TrainingExample(inputs={"count": 42}, reference="valid"),
    ]

    manifest = await optimizer.compile(agent, trainset, [])

    assert len(manifest.few_shot_examples) == 1
    assert manifest.few_shot_examples[0].inputs["count"] == 42


@pytest.mark.asyncio
async def test_bootstrap_list_reference() -> None:
    """Test that mining works when reference is a list of valid options."""
    llm = ComplexMockLLMClient()
    metric = ExactMatch()
    config = OptimizerConfig()
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = GenericMockAgent()
    # Reference allows "red" or "blue"
    trainset = [
        TrainingExample(inputs={"q": "color"}, reference=["red", "blue"]),
    ]

    manifest = await optimizer.compile(agent, trainset, [])

    # LLM returns "red", which matches one of the options
    assert len(manifest.few_shot_examples) == 1
    assert manifest.few_shot_examples[0].reference == ["red", "blue"]


@pytest.mark.asyncio
async def test_bootstrap_multiline_system_prompt() -> None:
    """Test that multiline system prompts are handled correctly."""
    llm = ComplexMockLLMClient()
    metric = ExactMatch()
    config = OptimizerConfig()
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = GenericMockAgent(system_prompt="Line 1\nLine 2")
    trainset = [
        TrainingExample(inputs={"q": "multi"}, reference="yes"),
    ]

    manifest = await optimizer.compile(agent, trainset, [])

    assert len(manifest.few_shot_examples) == 1
    assert manifest.optimized_instruction == "Line 1\nLine 2"


@pytest.mark.asyncio
async def test_bootstrap_duplicate_mining() -> None:
    """Test behavior when multiple identical examples succeed."""

    # We use a simple client for this
    class EchoLLMClient:
        async def generate(
            self,
            messages: list[dict[str, str]],
            model: str | None = None,
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> LLMResponse:
            return LLMResponse(content="4", usage=UsageStats())

    llm = EchoLLMClient()
    metric = ExactMatch()
    config = OptimizerConfig(max_bootstrapped_demos=5)  # Allow enough
    optimizer = BootstrapFewShot(llm_client=llm, metric=metric, config=config)

    agent = GenericMockAgent()
    # Duplicate examples in trainset
    trainset = [
        TrainingExample(inputs={"q": "2+2"}, reference="4"),
        TrainingExample(inputs={"q": "2+2"}, reference="4"),
    ]

    manifest = await optimizer.compile(agent, trainset, [])

    # Current implementation does not deduplicate, so we expect 2
    assert len(manifest.few_shot_examples) == 2
    assert manifest.few_shot_examples[0].inputs == {"q": "2+2"}
    assert manifest.few_shot_examples[1].inputs == {"q": "2+2"}
