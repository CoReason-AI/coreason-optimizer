# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from unittest.mock import Mock

from coreason_optimizer.strategies.mutator import IdentityMutator


def test_identity_mutator() -> None:
    mock_llm = Mock()
    mutator = IdentityMutator(llm_client=mock_llm)

    instruction = "You are a helpful assistant."
    new_instruction = mutator.mutate(instruction)

    assert new_instruction == instruction

    new_instruction_with_failures = mutator.mutate(instruction, failed_examples=["fail1"])
    assert new_instruction_with_failures == instruction
