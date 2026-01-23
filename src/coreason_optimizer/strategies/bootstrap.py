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
BootstrapFewShot Optimization Strategy.

This strategy improves agent performance by mining successful traces from
the training set (where the model got the answer right) and using them as
few-shot examples in the final prompt.
"""

import uuid
from typing import Any

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import (
    Construct,
    LLMClient,
    Metric,
    PromptOptimizer,
)
from coreason_optimizer.core.models import OptimizedManifest, TrainingExample
from coreason_optimizer.utils.logger import logger


class BootstrapFewShot(PromptOptimizer):
    """
    BootstrapFewShot strategy implementation.

    Process:
    1. Iterate through the training set.
    2. Attempt to solve each example using the current system prompt (zero-shot).
    3. Verify the prediction against the ground truth using the metric.
    4. If successful, collect the example as a candidate for few-shot learning.
    5. Select the best candidates to include in the final manifest.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        metric: Metric,
        config: OptimizerConfig,
    ):
        """
        Initialize the BootstrapFewShot optimizer.

        Args:
            llm_client: The LLM client to use for generation.
            metric: The metric to verify correctness.
            config: Optimization configuration.
        """
        self.llm_client = llm_client
        self.metric = metric
        self.config = config

    def _format_prompt(
        self,
        system_prompt: str,
        examples: list[TrainingExample],
        inputs: dict[str, Any],
    ) -> str:
        """
        Sensible default prompt formatter.
        Structure:
        ### System Instruction
        ...
        ### Examples
        Input: ...
        Output: ...
        ### User Input
        Input: ...
        """
        parts = []

        # System Prompt
        parts.append(f"### System Instruction\n{system_prompt}")

        # Examples
        if examples:
            parts.append("### Examples")
            for ex in examples:
                # We assume inputs are dicts, we serialize them simply
                input_str = ", ".join(f"{k}: {v}" for k, v in ex.inputs.items())
                parts.append(f"Input: {input_str}\nOutput: {ex.reference}")

        # User Input
        parts.append("### User Input")
        current_input_str = ", ".join(f"{k}: {v}" for k, v in inputs.items())
        parts.append(f"Input: {current_input_str}")

        return "\n\n".join(parts)

    def compile(
        self,
        agent: Construct,
        trainset: list[TrainingExample],
        valset: list[TrainingExample],
    ) -> OptimizedManifest:
        """
        Run the bootstrapping loop.

        Args:
            agent: The agent construct to optimize.
            trainset: Training examples to mine.
            valset: Validation examples for final scoring.

        Returns:
            An OptimizedManifest containing the best few-shot examples found.

        Raises:
            BudgetExceededError: If the budget limit is reached.
        """
        logger.info(
            "Starting BootstrapFewShot compilation",
            train_size=len(trainset),
            target_model=self.config.target_model,
        )

        successful_traces: list[TrainingExample] = []

        # 1. Mine successful traces
        for i, example in enumerate(trainset):
            # Format prompt with *no* examples initially (zero-shot) to see if the model can solve it
            # Or should we use existing examples? The prompt implies "BootstrapFewShot" mines traces.
            # Usually we start with 0-shot to find easy examples that become 1-shot for others.
            prompt = self._format_prompt(
                system_prompt=agent.system_prompt,
                examples=[],
                inputs=example.inputs,
            )

            # 2. Generate
            try:
                response = self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config.target_model,
                    temperature=0.0,  # Deterministic for mining
                )
            except Exception as e:
                logger.error(f"Error generating for example {i}: {e}")
                continue

            prediction = response.content

            # 3. Score
            score = self.metric(prediction, example.reference)

            # 4. Filter
            # exact_match returns 1.0 or 0.0. F1 returns 0.0-1.0.
            # We treat strict 1.0 as success for now, or maybe >= threshold?
            # Given PRD examples (ExactMatch), 1.0 is safe.
            if score >= 1.0:
                logger.debug(f"Example {i} passed with score {score}")
                successful_traces.append(example)
            else:
                logger.debug(f"Example {i} failed with score {score}")

        # 5. Select Candidates
        # We take up to max_bootstrapped_demos
        num_demos = min(len(successful_traces), self.config.max_bootstrapped_demos)
        selected_examples = successful_traces[:num_demos]

        logger.info(f"Bootstrapping complete. Selected {len(selected_examples)} examples.")

        # 6. Evaluate on Validation Set (to get performance_metric)
        # We run the AGENT (now with the selected examples) on the valset
        total_score = 0.0
        if valset:
            for example in valset:
                prompt = self._format_prompt(
                    system_prompt=agent.system_prompt,
                    examples=selected_examples,
                    inputs=example.inputs,
                )
                try:
                    response = self.llm_client.generate(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.config.target_model,
                        temperature=0.0,
                    )
                    s = self.metric(response.content, example.reference)
                    total_score += s
                except Exception:
                    pass
            avg_score = total_score / len(valset)
        else:
            avg_score = 0.0

        # 7. Create Manifest
        return OptimizedManifest(
            agent_id="unknown_agent",  # Agent protocol doesn't have ID?
            base_model=self.config.target_model,
            optimized_instruction=agent.system_prompt,  # Instruction is unchanged in Bootstrap
            few_shot_examples=selected_examples,
            performance_metric=avg_score,
            optimization_run_id=f"opt_{uuid.uuid4().hex[:8]}",
        )
