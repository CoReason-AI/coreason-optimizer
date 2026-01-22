# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import MagicMock, patch

import pytest

from coreason_optimizer.core.budget import BudgetExceededError
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import Construct, LLMClient, LLMResponse
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mipro import MiproOptimizer


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.generate.return_value = LLMResponse(content="Response", usage={"total_tokens": 10}, cost_usd=0.001)
    return client


@pytest.fixture
def mock_metric() -> MagicMock:
    return MagicMock(return_value=1.0)


@pytest.fixture
def config() -> OptimizerConfig:
    return OptimizerConfig(target_model="gpt-4", meta_model="gpt-4")


@pytest.fixture
def agent() -> Construct:
    class Agent(Construct):
        system_prompt = "Sys"
        inputs = ["q"]
        outputs = ["a"]

    return Agent()


@pytest.fixture
def trainset() -> list[TrainingExample]:
    return [TrainingExample(inputs={"q": "q1"}, reference="a1")]


def test_mipro_init_semantic_requires_embedding(mock_client: MagicMock, mock_metric: MagicMock) -> None:
    conf = OptimizerConfig(selector_type="semantic")
    with pytest.raises(ValueError, match="Embedding provider is required"):
        MiproOptimizer(mock_client, mock_metric, conf)


def test_mipro_diagnosis_budget_exceeded(
    mock_client: MagicMock,
    mock_metric: MagicMock,
    config: OptimizerConfig,
    agent: Construct,
    trainset: list[TrainingExample],
) -> None:
    mock_client.generate.side_effect = BudgetExceededError("Budget")
    optimizer = MiproOptimizer(mock_client, mock_metric, config)

    with pytest.raises(BudgetExceededError):
        optimizer.compile(agent, trainset, [])


def test_mipro_diagnosis_failure_logging(
    mock_client: MagicMock,
    mock_metric: MagicMock,
    config: OptimizerConfig,
    agent: Construct,
    trainset: list[TrainingExample],
) -> None:
    # Simulate a generic error during diagnosis
    mock_client.generate.side_effect = Exception("API Error")
    optimizer = MiproOptimizer(mock_client, mock_metric, config)

    # We patch mutator to avoid network calls there too
    with patch("coreason_optimizer.strategies.mipro.LLMInstructionMutator") as MockMutator:
        mock_mutator_inst = MagicMock()
        mock_mutator_inst.mutate.return_value = "New Instr"
        MockMutator.return_value = mock_mutator_inst

        manifest = optimizer.compile(agent, trainset, [])
        assert manifest.optimized_instruction == "Sys"  # Default if nothing better found


def test_mipro_candidate_generation_budget_exceeded(
    mock_client: MagicMock,
    mock_metric: MagicMock,
    config: OptimizerConfig,
    agent: Construct,
    trainset: list[TrainingExample],
) -> None:
    optimizer = MiproOptimizer(mock_client, mock_metric, config)

    # Diagnosis succeeds
    mock_client.generate.return_value = LLMResponse(content="wrong", usage={}, cost_usd=0.0)
    mock_metric.return_value = 0.0  # Force failure

    # Let's manually replace the mutator on the instance using patch.object
    with patch.object(optimizer.mutator, "mutate", side_effect=BudgetExceededError("Budget")):
        with pytest.raises(BudgetExceededError):
            optimizer.compile(agent, trainset, [])


def test_mipro_candidate_generation_failure(
    mock_client: MagicMock,
    mock_metric: MagicMock,
    config: OptimizerConfig,
    agent: Construct,
    trainset: list[TrainingExample],
) -> None:
    optimizer = MiproOptimizer(mock_client, mock_metric, config)

    # Diagnosis succeeds
    mock_client.generate.return_value = LLMResponse(content="wrong", usage={}, cost_usd=0.0)
    mock_metric.return_value = 0.0

    # Mutator raises generic error
    with patch.object(optimizer.mutator, "mutate", side_effect=Exception("Mutator Error")):
        # Should continue and use base instruction
        manifest = optimizer.compile(agent, trainset, [])
        assert manifest.optimized_instruction == "Sys"


def test_mipro_grid_search_budget_exceeded(
    mock_client: MagicMock,
    mock_metric: MagicMock,
    config: OptimizerConfig,
    agent: Construct,
    trainset: list[TrainingExample],
) -> None:
    # Mock mutator to return 1 candidate
    optimizer = MiproOptimizer(mock_client, mock_metric, config)

    with patch.object(optimizer.mutator, "mutate", return_value="New Instr"):
        # Diagnosis succeeds
        # We need to control the generate call.
        # The FIRST calls are diagnosis (1 call per trainset example).
        # Then mutation calls (controlled by mutator mock - we mocked mutate directly).
        # Then grid search calls.

        call_count = 0

        def side_effect(*args, **kwargs):  # type: ignore
            nonlocal call_count
            call_count += 1
            if call_count > 1:  # Fail after diagnosis (on grid search eval)
                raise BudgetExceededError("Budget")
            return LLMResponse(content="resp", usage={}, cost_usd=0.0)

        mock_client.generate.side_effect = side_effect

        with pytest.raises(BudgetExceededError):
            optimizer.compile(agent, trainset, [])


def test_mipro_evaluate_candidate_generic_exception(
    mock_client: MagicMock,
    mock_metric: MagicMock,
    config: OptimizerConfig,
    agent: Construct,
    trainset: list[TrainingExample],
) -> None:
    optimizer = MiproOptimizer(mock_client, mock_metric, config)

    # Force evaluate_candidate to raise exception
    mock_client.generate.side_effect = Exception("Grid Search Error")

    # Just check it returns 0.0 score essentially (doesn't crash)
    score = optimizer._evaluate_candidate("Instr", [], trainset)
    assert score == 0.0
