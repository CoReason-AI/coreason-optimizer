# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import AsyncMock, MagicMock

import pytest

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mutator import IdentityMutator, LLMInstructionMutator


@pytest.mark.asyncio
async def test_identity_mutator() -> None:
    mock_llm = MagicMock()
    mutator = IdentityMutator(llm_client=mock_llm)

    instruction = "You are a helpful assistant."
    new_instruction = await mutator.mutate(instruction)

    assert new_instruction == instruction

    # Updated to match new signature
    failed_example = TrainingExample(
        inputs={"q": "fail"},
        reference="answer",
    )
    new_instruction_with_failures = await mutator.mutate(instruction, failed_examples=[failed_example])
    assert new_instruction_with_failures == instruction


@pytest.mark.asyncio
async def test_llm_instruction_mutator_no_failures() -> None:
    mock_llm = AsyncMock()
    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    instruction = "Original instruction."
    # No failed examples -> return original
    new_instruction = await mutator.mutate(instruction, failed_examples=[])
    assert new_instruction == instruction

    new_instruction_none = await mutator.mutate(instruction, failed_examples=None)
    assert new_instruction_none == instruction

    mock_llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_llm_instruction_mutator_success() -> None:
    mock_llm = AsyncMock()
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

    new_instruction = await mutator.mutate(instruction, failed_examples=[failed_example])

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


@pytest.mark.asyncio
async def test_llm_instruction_mutator_cleanup() -> None:
    """Test that markdown code blocks are stripped."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "```markdown\nClean instruction.\n```"
    mock_llm.generate.return_value = mock_response

    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    # We need failures to trigger the call
    failed_example = TrainingExample(inputs={"q": "f"}, reference="a")

    new_instruction = await mutator.mutate("instr", failed_examples=[failed_example])

    assert new_instruction == "Clean instruction."


@pytest.mark.asyncio
async def test_llm_instruction_mutator_failure() -> None:
    """Test when the meta-LLM call fails."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = Exception("API Error")

    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    instruction = "Do task."
    failed_example = TrainingExample(inputs={"q": "fail"}, reference="a")

    new_instruction = await mutator.mutate(instruction, failed_examples=[failed_example])

    # Should fall back to original instruction
    assert new_instruction == instruction


@pytest.mark.asyncio
async def test_llm_instruction_mutator_empty_response() -> None:
    """Test fallback when LLM returns empty string."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "   "  # Whitespace only
    mock_llm.generate.return_value = mock_response

    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)
    failed_example = TrainingExample(inputs={"q": "f"}, reference="a")

    new_instruction = await mutator.mutate("original", failed_examples=[failed_example])
    assert new_instruction == "original"


@pytest.mark.asyncio
async def test_llm_instruction_mutator_context_limit() -> None:
    """Test that failed examples are truncated in the prompt."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "new"
    mock_llm.generate.return_value = mock_response

    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    # Create 20 examples
    failures = [TrainingExample(inputs={"id": i}, reference=f"ref{i}") for i in range(20)]

    await mutator.mutate("instr", failed_examples=failures)

    # Check prompt content
    args, kwargs = mock_llm.generate.call_args
    prompt = kwargs["messages"][0]["content"]

    assert "Example 1:" in prompt
    assert "Example 10:" in prompt
    assert "Example 11:" not in prompt
    assert "... (and 10 more failures)" in prompt


@pytest.mark.asyncio
async def test_llm_instruction_mutator_complex_inputs() -> None:
    """Test handling of complex nested input structures."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "new"
    mock_llm.generate.return_value = mock_response

    config = OptimizerConfig()
    mutator = LLMInstructionMutator(llm_client=mock_llm, config=config)

    complex_input = {
        "user_profile": {"age": 30, "tags": ["a", "b"]},
        "history": [{"role": "user", "text": "hi"}],
    }
    failed_example = TrainingExample(inputs=complex_input, reference="resp")

    await mutator.mutate("instr", failed_examples=[failed_example])

    args, kwargs = mock_llm.generate.call_args
    prompt = kwargs["messages"][0]["content"]

    # Verify string representation shows up
    assert '"user_profile":' in prompt
    assert '"tags":' in prompt
    assert '"history":' in prompt
    assert '"age": 30' in prompt
