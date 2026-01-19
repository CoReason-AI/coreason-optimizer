# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from abc import ABC, abstractmethod

from coreason_optimizer.core.interfaces import LLMClient


class BaseMutator(ABC):
    """Abstract base class for instruction mutation strategies."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    @abstractmethod
    def mutate(self, current_instruction: str, failed_examples: list[str] | None = None) -> str:
        """Generate a new instruction based on the current one and optional failure cases."""
        pass  # pragma: no cover


class IdentityMutator(BaseMutator):
    """A mutator that returns the instruction unchanged. Useful for baselines."""

    def mutate(self, current_instruction: str, failed_examples: list[str] | None = None) -> str:
        return current_instruction
