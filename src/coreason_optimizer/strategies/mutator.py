# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

"""
Instruction Mutation Strategy.

This module provides the logic for using a Meta-LLM to rewrite system instructions
based on observed failure cases.
"""

import json
from abc import ABC, abstractmethod

from jinja2 import Template

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMClient
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.utils.exceptions import BudgetExceededError
from coreason_optimizer.utils.logger import logger

META_PROMPT_TEMPLATE = """
You are an expert prompt engineer. Your goal is to improve the following system instruction
based on the provided failure cases.

Current Instruction:
"{{ instruction }}"

The instruction failed on the following examples:
{% for failure in failures %}
Example {{ loop.index }}:
Input: {{ failure.inputs }}
Expected Output: {{ failure.reference }}
Actual Output: {{ failure.prediction }}
{% endfor %}
{% if failures_hidden_count > 0 %}
... and {{ failures_hidden_count }} more failures.
{% endif %}

Please rewrite the system instruction to address these failures while maintaining
performance on general cases. Return ONLY the new instruction text.
"""


class BaseMutator(ABC):
    """Abstract base class for instruction mutation strategies."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the BaseMutator.

        Args:
            llm_client: The LLM client to use for mutation.
        """
        self.llm_client = llm_client

    @abstractmethod
    def mutate(
        self,
        current_instruction: str,
        failed_examples: list[TrainingExample] | None = None,
    ) -> str:
        """
        Generate a new instruction based on the current one and optional failure cases.

        Args:
            current_instruction: The existing system prompt.
            failed_examples: A list of examples that the current instruction failed on.

        Returns:
            The new system instruction string.
        """
        pass  # pragma: no cover


class IdentityMutator(BaseMutator):
    """A mutator that returns the instruction unchanged. Useful for baselines."""

    def mutate(
        self,
        current_instruction: str,
        failed_examples: list[TrainingExample] | None = None,
    ) -> str:
        """
        Return the instruction as-is.

        Args:
            current_instruction: The instruction.
            failed_examples: Ignored.

        Returns:
            The same instruction.
        """
        return current_instruction


class LLMInstructionMutator(BaseMutator):
    """Mutates instructions using a Meta-LLM to address failures."""

    def __init__(self, llm_client: LLMClient, config: OptimizerConfig):
        """
        Initialize the LLMInstructionMutator.

        Args:
            llm_client: The LLM client for the meta-prompt.
            config: Configuration object (e.g., for meta_model name).
        """
        super().__init__(llm_client)
        self.config = config

    def mutate(
        self,
        current_instruction: str,
        failed_examples: list[TrainingExample] | None = None,
    ) -> str:
        """
        Generate a new instruction by asking the Meta-LLM to analyze failures.

        Args:
            current_instruction: The current instruction.
            failed_examples: List of TrainingExample where the current instruction failed.

        Returns:
            A new, potentially improved instruction string.
        """
        if not failed_examples:
            logger.warning("No failed examples provided for mutation. Returning original instruction.")
            return current_instruction

        meta_prompt = self._build_meta_prompt(current_instruction, failed_examples)

        try:
            logger.info("Requesting instruction mutation from Meta-LLM.")
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": meta_prompt}],
                model=self.config.meta_model,
                temperature=0.7,
            )
            new_instruction = response.content.strip()
            # Basic cleanup if the model wraps it in quotes or markdown
            if new_instruction.startswith("```") and new_instruction.endswith("```"):
                lines = new_instruction.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                new_instruction = "\n".join(lines).strip()

            if not new_instruction:
                logger.warning("Meta-LLM returned empty instruction. Returning original.")
                return current_instruction

            return new_instruction
        except BudgetExceededError:
            raise
        except Exception as e:
            logger.error(f"Failed to mutate instruction: {e}")
            return current_instruction

    def _build_meta_prompt(self, instruction: str, failures: list[TrainingExample]) -> str:
        """
        Construct the meta-prompt for the LLM using Jinja2.

        Args:
            instruction: Current instruction.
            failures: List of failure examples.

        Returns:
            The full prompt string for the Meta-LLM.
        """
        display_failures = failures[:10]
        failures_hidden_count = len(failures) - len(display_failures)

        formatted_failures = []
        for ex in display_failures:
            formatted_failures.append(
                {
                    "inputs": json.dumps(ex.inputs, indent=2),
                    "reference": str(ex.reference),
                    "prediction": str(ex.metadata.get("prediction", "N/A")),
                }
            )

        template = Template(META_PROMPT_TEMPLATE)
        return str(
            template.render(
                instruction=instruction,
                failures=formatted_failures,
                failures_hidden_count=failures_hidden_count,
            )
        )
