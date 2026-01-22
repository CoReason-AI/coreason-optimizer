# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import MagicMock

import pytest

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMClient, LLMResponse
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.strategies.mutator import IdentityMutator, LLMInstructionMutator
from coreason_optimizer.utils.exceptions import BudgetExceededError


def test_identity_mutator() -> None:
    mutator = IdentityMutator(llm_client=MagicMock())
    assert mutator.mutate("instr", None) == "instr"
    assert mutator.mutate("instr", []) == "instr"


def test_llm_mutator_no_failures() -> None:
    """Test returning original instruction if no failures provided."""
    mock_client = MagicMock(spec=LLMClient)
    config = OptimizerConfig()
    mutator = LLMInstructionMutator(mock_client, config)

    res = mutator.mutate("original", [])
    assert res == "original"
    mock_client.generate.assert_not_called()


def test_llm_mutator_success() -> None:
    """Test successful mutation."""
    mock_client = MagicMock(spec=LLMClient)
    # The client.generate is called with messages=[{"role": "user", "content": ...}]
    mock_client.generate.return_value = LLMResponse(
        content="new instruction", usage={"total_tokens": 10}, cost_usd=0.001
    )

    config = OptimizerConfig(meta_model="gpt-4")
    mutator = LLMInstructionMutator(mock_client, config)

    failures = [TrainingExample(inputs={"q": "fail"}, reference="ref")]

    new_instr = mutator.mutate("original", failures)

    assert new_instr == "new instruction"
    mock_client.generate.assert_called_once()

    # Check call arguments.
    # If called as keyword args:
    kwargs = mock_client.generate.call_args.kwargs
    assert kwargs["model"] == "gpt-4"
    assert "messages" in kwargs
    content = kwargs["messages"][0]["content"]
    assert "original" in content
    assert "fail" in content


def test_llm_mutator_cleanup_markdown() -> None:
    """Test cleaning up markdown code blocks from response."""
    mock_client = MagicMock(spec=LLMClient)
    content = "```\ncleaned instruction\n```"
    mock_client.generate.return_value = LLMResponse(content=content, usage={}, cost_usd=0.0)

    mutator = LLMInstructionMutator(mock_client, OptimizerConfig())
    failures = [TrainingExample(inputs={"q": "fail"}, reference="ref")]

    new_instr = mutator.mutate("original", failures)
    assert new_instr == "cleaned instruction"


def test_llm_mutator_empty_response() -> None:
    """Test handling empty response (fallback to original)."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate.return_value = LLMResponse(content="", usage={}, cost_usd=0.0)

    mutator = LLMInstructionMutator(mock_client, OptimizerConfig())
    failures = [TrainingExample(inputs={"q": "fail"}, reference="ref")]

    new_instr = mutator.mutate("original", failures)
    assert new_instr == "original"


def test_llm_mutator_budget_exceeded() -> None:
    """Test re-raising BudgetExceededError."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate.side_effect = BudgetExceededError("Budget exceeded")

    mutator = LLMInstructionMutator(mock_client, OptimizerConfig())
    failures = [TrainingExample(inputs={"q": "fail"}, reference="ref")]

    with pytest.raises(BudgetExceededError):
        mutator.mutate("original", failures)


def test_llm_mutator_generic_exception() -> None:
    """Test catching generic exceptions and returning original."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate.side_effect = Exception("API Error")

    mutator = LLMInstructionMutator(mock_client, OptimizerConfig())
    failures = [TrainingExample(inputs={"q": "fail"}, reference="ref")]

    new_instr = mutator.mutate("original", failures)
    assert new_instr == "original"
