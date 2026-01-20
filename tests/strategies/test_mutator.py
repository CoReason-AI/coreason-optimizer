# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import MagicMock, Mock

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mutator import IdentityMutator, LLMInstructionMutator


def test_identity_mutator() -> None:
    mock_llm = Mock()
    mutator = IdentityMutator(llm_client=mock_llm)

    instruction = "You are a helpful assistant."
    new_instruction = mutator.mutate(instruction)

    assert new_instruction == instruction

    # Updated to match new signature
    failed_example = TrainingExample(
        inputs={"q": "fail"},
        reference="answer",
    )
    new_instruction_with_failures = mutator.mutate(instruction, failed_examples=[failed_example])
    assert new_instruction_with_failures == instruction


def test_llm_instruction_mutator_no_failures() -> None:
    mock_llm = Mock()
    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    instruction = "Original instruction."
    # No failed examples -> return original
    new_instruction = mutator.mutate(instruction, failed_examples=[])
    assert new_instruction == instruction

    new_instruction_none = mutator.mutate(instruction, failed_examples=None)
    assert new_instruction_none == instruction

    mock_llm.generate.assert_not_called()


def test_llm_instruction_mutator_success() -> None:
    mock_llm = Mock()
    # Mock response
    mock_response = MagicMock()
    mock_response.content = "Improved instruction."
    mock_llm.generate.return_value = mock_response

    config = OptimizerConfig(meta_model="gpt-4o")
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    instruction = "Do task."
    failed_example = TrainingExample(
        inputs={"input": "hard case"}, reference="correct output", metadata={"prediction": "wrong output"}
    )

    new_instruction = mutator.mutate(instruction, failed_examples=[failed_example])

    assert new_instruction == "Improved instruction."

    # Verify the prompt sent to LLM
    mock_llm.generate.assert_called_once()
    call_kwargs = mock_llm.generate.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]

    assert "Current Instruction" in content
    assert "Do task." in content
    assert "Failure Analysis" in content
    assert "hard case" in content
    assert "correct output" in content
    assert "wrong output" in content

    assert call_kwargs["model"] == "gpt-4o"


def test_llm_instruction_mutator_cleanup() -> None:
    """Test that markdown code blocks are stripped."""
    mock_llm = Mock()
    mock_response = MagicMock()
    mock_response.content = "```markdown\nClean instruction.\n```"
    mock_llm.generate.return_value = mock_response

    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    # We need failures to trigger the call
    failed_example = TrainingExample(inputs={"q": "f"}, reference="a")

    new_instruction = mutator.mutate("instr", failed_examples=[failed_example])

    assert new_instruction == "Clean instruction."


def test_llm_instruction_mutator_failure() -> None:
    """Test when the meta-LLM call fails."""
    mock_llm = Mock()
    mock_llm.generate.side_effect = Exception("API Error")

    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    instruction = "Do task."
    failed_example = TrainingExample(inputs={"q": "fail"}, reference="a")

    new_instruction = mutator.mutate(instruction, failed_examples=[failed_example])

    # Should fall back to original instruction
    assert new_instruction == instruction
