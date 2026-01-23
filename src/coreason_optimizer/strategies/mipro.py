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
MIPRO (Multi-prompt Instruction PRoposal Optimizer) Strategy.

This advanced optimization strategy combines instruction mutation (via a Meta-LLM)
and few-shot example selection to find the optimal prompt configuration.
"""

import uuid

from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.core.client import (
    BudgetAwareEmbeddingProvider,
    BudgetAwareLLMClient,
)
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.formatter import format_prompt
from coreason_optimizer.core.interfaces import (
    Construct,
    EmbeddingProvider,
    LLMClient,
    Metric,
    PromptOptimizer,
)
from coreason_optimizer.core.models import OptimizedManifest, TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.mutator import LLMInstructionMutator
from coreason_optimizer.strategies.selector import (
    BaseSelector,
    RandomSelector,
    SemanticSelector,
)
from coreason_optimizer.utils.exceptions import BudgetExceededError
from coreason_optimizer.utils.logger import logger


class MiproOptimizer(PromptOptimizer):
    """
    MIPRO (Multi-prompt Instruction PRoposal Optimizer) Strategy.

    This strategy:
    1. Generates N candidate system instructions using a mutator (Meta-LLM).
    2. Generates M candidate few-shot example sets using a selector.
    3. Performs a grid search over all (Instruction, ExampleSet) combinations.
    4. Selects the combination with the highest score on the training set.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        metric: Metric,
        config: OptimizerConfig,
        embedding_provider: EmbeddingProvider | None = None,
        num_instruction_candidates: int = 10,
        num_fewshot_combinations: int = 5,
    ):
        """
        Initialize the MIPRO Optimizer.

        Args:
            llm_client: The LLM client for generation.
            metric: The metric for evaluation.
            config: Optimization configuration.
            embedding_provider: Optional provider for semantic selection.
            num_instruction_candidates: Number of instruction variations to generate.
            num_fewshot_combinations: Number of few-shot sets to sample.

        Raises:
            ValueError: If semantic selection is requested but no embedding provider is given.
        """
        self.metric = metric
        self.config = config
        self.num_instruction_candidates = num_instruction_candidates
        self.num_fewshot_combinations = num_fewshot_combinations

        # Wrap client with Budget Awareness
        self.budget_manager = BudgetManager(config.budget_limit_usd)
        self.llm_client = BudgetAwareLLMClient(llm_client, self.budget_manager)

        # Initialize components
        self.mutator = LLMInstructionMutator(self.llm_client, config)

        self.selector: BaseSelector
        if config.selector_type == "semantic":
            if not embedding_provider:
                raise ValueError("Embedding provider is required for semantic selection.")

            # Wrap embedding provider sharing the SAME budget manager
            wrapped_embedder = BudgetAwareEmbeddingProvider(embedding_provider, self.budget_manager)
            self.selector = SemanticSelector(wrapped_embedder, seed=42, embedding_model=config.embedding_model)
        else:
            self.selector = RandomSelector(seed=42)

    def _evaluate_candidate(
        self,
        instruction: str,
        examples: list[TrainingExample],
        dataset: list[TrainingExample],
    ) -> float:
        """Evaluate a single candidate (instruction + examples) on a dataset."""
        total_score = 0.0
        for example in dataset:
            prompt = format_prompt(instruction, examples, example.inputs)
            try:
                response = self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config.target_model,
                    temperature=0.0,
                )
                score = self.metric(response.content, example.reference)
                total_score += score
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.warning(f"Error during evaluation: {e}")
                pass

        return total_score / len(dataset) if dataset else 0.0

    def compile(
        self,
        agent: Construct,
        trainset: list[TrainingExample],
        valset: list[TrainingExample],
    ) -> OptimizedManifest:
        """
        Run the MIPRO optimization loop.

        Args:
            agent: The agent construct.
            trainset: Training data.
            valset: Validation data.

        Returns:
            OptimizedManifest with best instruction and examples.

        Raises:
            BudgetExceededError: If budget is exceeded.
        """
        logger.info(
            "Starting MIPRO compilation",
            train_size=len(trainset),
            target_model=self.config.target_model,
        )

        # 1. Diagnosis: Run baseline to find failures
        logger.info("Running baseline diagnosis...")
        dataset_obj = Dataset(trainset)
        failed_examples = []

        # We need to run at least once to get failures.
        # We use the original instruction and NO examples (or random examples?) for diagnosis.
        # Let's use 0-shot with original instruction.
        for example in trainset:
            prompt = format_prompt(agent.system_prompt, [], example.inputs)
            try:
                response = self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config.target_model,
                    temperature=0.0,
                )
                score = self.metric(response.content, example.reference)
                if score < 1.0:  # Assuming < 1.0 is failure/imperfect
                    # We store the *prediction* in metadata for the mutator
                    example.metadata["prediction"] = response.content
                    failed_examples.append(example)
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.error(f"Error diagnosing example: {e}")

        logger.info(f"Diagnosis complete. Found {len(failed_examples)} failures.")

        # 2. Candidate Generation: Instructions
        instruction_candidates = {agent.system_prompt}  # Use set to avoid duplicates
        logger.info(f"Generating {self.num_instruction_candidates} instruction candidates...")

        for i in range(self.num_instruction_candidates):
            try:
                new_instruction = self.mutator.mutate(
                    current_instruction=agent.system_prompt,
                    failed_examples=failed_examples,
                )
                instruction_candidates.add(new_instruction)
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.warning(f"Failed to generate instruction candidate {i}: {e}")

        instruction_list = list(instruction_candidates)
        logger.info(f"Generated {len(instruction_list)} unique instruction candidates.")

        # 3. Candidate Generation: Example Sets
        example_sets: list[list[TrainingExample]] = []
        # Always include 0-shot
        example_sets.append([])

        logger.info(f"Generating {self.num_fewshot_combinations} few-shot sets...")
        for _ in range(self.num_fewshot_combinations):
            # Randomly select k examples (using max_bootstrapped_demos from config)
            k = self.config.max_bootstrapped_demos
            selected = self.selector.select(dataset_obj, k=k)
            example_sets.append(selected)

        # 4. Grid Search
        best_score = -1.0
        best_instruction = agent.system_prompt
        best_examples: list[TrainingExample] = []

        logger.info(
            f"Starting Grid Search: {len(instruction_list)} inst x {len(example_sets)} example sets "
            f"= {len(instruction_list) * len(example_sets)} candidates."
        )

        for instr in instruction_list:
            for ex_set in example_sets:
                # Evaluate on Trainset (Optimization Objective)
                # In production, we might want to evaluate on a held-out 'dev' split of trainset
                # to avoid overfitting, but for now we use the provided trainset.
                score = self._evaluate_candidate(instr, ex_set, trainset)

                logger.debug(f"Candidate Score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_instruction = instr
                    best_examples = ex_set

        logger.info(f"Grid Search complete. Best Training Score: {best_score}")

        # 5. Final Evaluation on Validation Set
        # If valset is provided, we compute the 'performance_metric' on it.
        # Otherwise we use the best training score.
        final_metric = best_score
        if valset:
            logger.info("Evaluating best candidate on Validation Set...")
            final_metric = self._evaluate_candidate(best_instruction, best_examples, valset)
            logger.info(f"Validation Score: {final_metric}")

        # 6. Create Manifest
        return OptimizedManifest(
            agent_id="unknown_agent",
            base_model=self.config.target_model,
            optimized_instruction=best_instruction,
            few_shot_examples=best_examples,
            performance_metric=final_metric,
            optimization_run_id=f"opt_mipro_{uuid.uuid4().hex[:8]}",
        )
