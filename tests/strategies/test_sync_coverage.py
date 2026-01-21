from unittest.mock import MagicMock

import pytest

from coreason_optimizer.core.budget import BudgetExceededError
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mutator import LLMInstructionMutator


@pytest.fixture
def mock_llm_client() -> MagicMock:
    return MagicMock()


def test_sync_mutator_success(mock_llm_client: MagicMock) -> None:
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutator(mock_llm_client, config)

    mock_llm_client.generate.return_value = LLMResponse(content="New Instruction", usage=UsageStats())

    # Test with failures
    failures = [TrainingExample(inputs={"q": "1"}, reference="A")]
    result = mutator.mutate("Original", failures)
    assert result == "New Instruction"

    # Test without failures (should return original)
    result_no_fail = mutator.mutate("Original", [])
    assert result_no_fail == "Original"


def test_sync_mutator_exception(mock_llm_client: MagicMock) -> None:
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutator(mock_llm_client, config)

    mock_llm_client.generate.side_effect = Exception("API Error")

    failures = [TrainingExample(inputs={"q": "1"}, reference="A")]
    # Should catch exception and return original
    result = mutator.mutate("Original", failures)
    assert result == "Original"


def test_sync_mutator_budget_exceeded(mock_llm_client: MagicMock) -> None:
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutator(mock_llm_client, config)

    mock_llm_client.generate.side_effect = BudgetExceededError("Budget exceeded")

    failures = [TrainingExample(inputs={"q": "1"}, reference="A")]
    # Should re-raise BudgetExceededError
    with pytest.raises(BudgetExceededError):
        mutator.mutate("Original", failures)


def test_sync_mutator_empty_response(mock_llm_client: MagicMock) -> None:
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutator(mock_llm_client, config)

    mock_llm_client.generate.return_value = LLMResponse(content="", usage=UsageStats())

    failures = [TrainingExample(inputs={"q": "1"}, reference="A")]
    # Should log warning and return original
    result = mutator.mutate("Original", failures)
    assert result == "Original"


def test_sync_mutator_markdown_cleanup(mock_llm_client: MagicMock) -> None:
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutator(mock_llm_client, config)

    mock_llm_client.generate.return_value = LLMResponse(content="```\nNew Instruction\n```", usage=UsageStats())

    failures = [TrainingExample(inputs={"q": "1"}, reference="A")]
    result = mutator.mutate("Original", failures)
    assert result == "New Instruction"
