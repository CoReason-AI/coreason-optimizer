# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import uuid

from coreason_optimizer.core.budget import BudgetExceededError, BudgetManager
from coreason_optimizer.core.client import BudgetAwareLLMClient
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.formatter import format_prompt
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
    BootstrapFewShot strategy:
    Mines the teacher model's successful traces on the training set to create
    few-shot examples for the optimized prompt.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        metric: Metric,
        config: OptimizerConfig,
    ):
        self.metric = metric
        self.config = config
        # Wrap client with Budget Awareness
        self.budget_manager = BudgetManager(config.budget_limit_usd)
        self.llm_client = BudgetAwareLLMClient(llm_client, self.budget_manager)

    def compile(
        self,
        agent: Construct,
        trainset: list[TrainingExample],
        valset: list[TrainingExample],
    ) -> OptimizedManifest:
        """
        Run the bootstrapping loop.
        1. Iterate over trainset.
        2. Run agent (Teacher mode) on input.
        3. Check if output matches reference via metric.
        4. If match, keep as candidate.
        5. Select top K candidates.
        6. Create manifest.
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
            prompt = format_prompt(
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
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.error(f"Error generating for example {i}: {e}")
                continue

            prediction = response.content

            # 3. Score
            score = self.metric(prediction, example.reference)

            # 4. Filter
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
                prompt = format_prompt(
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
