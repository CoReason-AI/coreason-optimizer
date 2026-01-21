from unittest.mock import MagicMock

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mutator import IdentityMutator, LLMInstructionMutator


def test_mutator_sync_direct_coverage() -> None:
    # Directly test the Sync LLMInstructionMutator class to ensure coverage

    mock_client = MagicMock()
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutator(mock_client, config)

    # Test case 1: Successful mutation
    mock_client.generate.return_value = LLMResponse(content="mutated", usage=UsageStats())
    result = mutator.mutate("original", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert result == "mutated"

    # Test case 2: Exception handling
    mock_client.generate.side_effect = ValueError("Error")
    result = mutator.mutate("original", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert result == "original"

    # Test case 3: Empty response
    mock_client.generate.side_effect = None
    mock_client.generate.return_value = LLMResponse(content="", usage=UsageStats())
    result = mutator.mutate("original", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert result == "original"

    # Test case 4: Markdown cleanup
    mock_client.generate.return_value = LLMResponse(content="```\nclean\n```", usage=UsageStats())
    result = mutator.mutate("original", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert result == "clean"


def test_identity_mutator_sync_coverage() -> None:
    mock_client = MagicMock()
    mutator = IdentityMutator(mock_client)
    result = mutator.mutate("original")
    assert result == "original"
