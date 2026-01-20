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

from coreason_optimizer.core.budget import BudgetExceededError
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMClient, LLMResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.bootstrap import BootstrapFewShot
from coreason_optimizer.strategies.mipro import MiproOptimizer


class CostlyMockClient(LLMClient):
    """A client that charges a fixed cost per call."""

    def __init__(self, cost_per_call: float):
        self.cost_per_call = cost_per_call

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="mock", usage=UsageStats(cost_usd=self.cost_per_call))


class MockConstruct:
    @property
    def system_prompt(self) -> str:
        return "You are a helpful assistant."

    @property
    def inputs(self) -> list[str]:
        return ["q"]

    @property
    def outputs(self) -> list[str]:
        return ["a"]


def test_bootstrap_budget_enforcement() -> None:
    """Test that BootstrapFewShot stops when budget is exceeded."""
    # Budget $1.0. Cost per call $0.6.
    # 1st call: $0.6. OK.
    # 2nd call: $1.2. Fail.

    config = OptimizerConfig(budget_limit_usd=1.0)
    client = CostlyMockClient(cost_per_call=0.6)

    # We pass a metric that always returns 1.0 so it keeps going
    def metric(prediction: str, reference: Any, **kwargs: Any) -> float:
        return 1.0

    optimizer = BootstrapFewShot(client, metric, config)

    # We provide 3 examples. It should fail on the 2nd one.
    trainset = [
        TrainingExample(inputs={"q": "1"}, reference="1"),
        TrainingExample(inputs={"q": "2"}, reference="2"),
        TrainingExample(inputs={"q": "3"}, reference="3"),
    ]

    with pytest.raises(BudgetExceededError):
        optimizer.compile(MockConstruct(), trainset, [])


def test_mipro_budget_enforcement_mutation() -> None:
    """Test that MiproOptimizer stops during mutation if budget exceeded."""
    config = OptimizerConfig(budget_limit_usd=1.0)
    client = CostlyMockClient(cost_per_call=0.6)

    # Diagnosis should fail (score 0.0) so we get failed examples
    def metric(prediction: str, reference: Any, **kwargs: Any) -> float:
        return 0.0

    optimizer = MiproOptimizer(client, metric, config)

    # 1 example.
    # Diagnosis: $0.6. Metric 0.0 -> Failed example.
    # Mutator: Called because we have failures.
    # Mutator call: $0.6. Total $1.2. Budget exceeded.

    trainset = [
        TrainingExample(inputs={"q": "1"}, reference="1"),
    ]

    with pytest.raises(BudgetExceededError):
        optimizer.compile(MockConstruct(), trainset, [])


def test_mipro_budget_enforcement_diagnosis() -> None:
    """Test that MiproOptimizer stops during diagnosis if budget exceeded."""
    config = OptimizerConfig(budget_limit_usd=1.0)
    client = CostlyMockClient(cost_per_call=0.6)

    def metric(prediction: str, reference: Any, **kwargs: Any) -> float:
        return 1.0

    optimizer = MiproOptimizer(client, metric, config)

    # 2 examples. Diagnosis will cost 0.6 * 2 = 1.2 > 1.0.
    trainset = [
        TrainingExample(inputs={"q": "1"}, reference="1"),
        TrainingExample(inputs={"q": "2"}, reference="2"),
    ]

    with pytest.raises(BudgetExceededError):
        optimizer.compile(MockConstruct(), trainset, [])


def test_mipro_budget_enforcement() -> None:
    """Test that MiproOptimizer stops when budget is exceeded."""
    # Budget $1.0. Cost per call $0.6.

    config = OptimizerConfig(budget_limit_usd=1.0)
    client = CostlyMockClient(cost_per_call=0.6)

    def metric(prediction: str, reference: Any, **kwargs: Any) -> float:
        return 1.0

    optimizer = MiproOptimizer(client, metric, config)

    trainset = [
        TrainingExample(inputs={"q": "1"}, reference="1"),
    ]

    # MIPRO flow:
    # 1. Diagnosis (1 call per example) -> $0.6. OK.
    # 2. Mutator (N candidates).
    #    Mutator calls Meta-LLM. That also costs $0.6.
    #    $0.6 (diagnosis) + $0.6 (mutation) = $1.2. Fail.

    with pytest.raises(BudgetExceededError):
        optimizer.compile(MockConstruct(), trainset, [])
