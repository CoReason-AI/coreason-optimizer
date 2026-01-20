# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import json
from abc import ABC, abstractmethod

from jinja2 import Template

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import LLMClient
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.utils.logger import logger

META_PROMPT_TEMPLATE = (
    "You are an expert prompt engineer. Your goal is to rewrite the system instruction "
    "for an AI agent to fix specific failure cases while maintaining general performance.\n\n"
    "### Current Instruction\n"
    "{{ instruction }}\n\n"
    "### Failure Analysis\n"
    "The following examples failed with the current instruction:\n"
    "{% for ex in failures %}\n"
    "Example {{ loop.index }}:\n"
    "Input:\n"
    "{{ ex.inputs }}\n\n"
    "Expected Output:\n"
    "{{ ex.reference }}\n\n"
    "Actual Output:\n"
    "{{ ex.prediction }}\n"
    "{% endfor %}\n"
    "{% if failures_hidden_count > 0 %}\n"
    "... (and {{ failures_hidden_count }} more failures)\n"
    "{% endif %}\n\n"
    "### Task\n"
    "Analyze the examples and the current instruction. Propose a NEW system instruction that would correctly handle "
    "these examples. Return ONLY the new instruction text, without explanation or markdown formatting."
)


class BaseMutator(ABC):
    """Abstract base class for instruction mutation strategies."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    @abstractmethod
    def mutate(
        self,
        current_instruction: str,
        failed_examples: list[TrainingExample] | None = None,
    ) -> str:
        """Generate a new instruction based on the current one and optional failure cases."""
        pass  # pragma: no cover


class IdentityMutator(BaseMutator):
    """A mutator that returns the instruction unchanged. Useful for baselines."""

    def mutate(
        self,
        current_instruction: str,
        failed_examples: list[TrainingExample] | None = None,
    ) -> str:
        return current_instruction


class LLMInstructionMutator(BaseMutator):
    """Mutates instructions using a Meta-LLM to address failures."""

    def __init__(self, llm_client: LLMClient, config: OptimizerConfig):
        super().__init__(llm_client)
        self.config = config

    def mutate(
        self,
        current_instruction: str,
        failed_examples: list[TrainingExample] | None = None,
    ) -> str:
        """
        Generate a new instruction by asking the Meta-LLM to analyze failures.
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
        except Exception as e:
            logger.error(f"Failed to mutate instruction: {e}")
            return current_instruction

    def _build_meta_prompt(self, instruction: str, failures: list[TrainingExample]) -> str:
        """Construct the meta-prompt for the LLM using Jinja2."""
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
