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
from typing import Any, Callable, cast

import anyio

from coreason_optimizer.core.async_client import (
    BudgetAwareEmbeddingProviderAsync,
    BudgetAwareLLMClientAsync,
)
from coreason_optimizer.core.budget import BudgetExceededError, BudgetManager
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.formatter import format_prompt
from coreason_optimizer.core.interfaces import (
    Construct,
    EmbeddingProvider,
    EmbeddingProviderAsync,
    LLMClient,
    LLMClientAsync,
    Metric,
    PromptOptimizer,
    PromptOptimizerAsync,
)
from coreason_optimizer.core.models import OptimizedManifest, TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.mutator import (
    LLMInstructionMutatorAsync,
)
from coreason_optimizer.strategies.selector import (
    BaseSelectorAsync,
    RandomSelectorAsync,
    SemanticSelectorAsync,
)
from coreason_optimizer.utils.adapters import SyncToAsyncEmbeddingProviderAdapter, SyncToAsyncLLMClientAdapter
from coreason_optimizer.utils.helpers import unwrap_exception_group
from coreason_optimizer.utils.logger import logger


class MiproOptimizerAsync(PromptOptimizerAsync):
    """
    Async MIPRO Strategy.
    """

    def __init__(
        self,
        llm_client: LLMClientAsync,
        metric: Metric,
        config: OptimizerConfig,
        embedding_provider: EmbeddingProviderAsync | None = None,
        num_instruction_candidates: int = 10,
        num_fewshot_combinations: int = 5,
    ):
        self.metric = metric
        self.config = config
        self.num_instruction_candidates = num_instruction_candidates
        self.num_fewshot_combinations = num_fewshot_combinations

        self.budget_manager = BudgetManager(config.budget_limit_usd)
        self.llm_client = BudgetAwareLLMClientAsync(llm_client, self.budget_manager)

        self.mutator = LLMInstructionMutatorAsync(self.llm_client, config)

        self.selector: BaseSelectorAsync
        if config.selector_type == "semantic":
            if not embedding_provider:
                raise ValueError("Embedding provider is required for semantic selection.")

            wrapped_embedder = BudgetAwareEmbeddingProviderAsync(embedding_provider, self.budget_manager)
            self.selector = SemanticSelectorAsync(wrapped_embedder, seed=42, embedding_model=config.embedding_model)
        else:
            self.selector = RandomSelectorAsync(seed=42)

    async def _evaluate_candidate(
        self,
        instruction: str,
        examples: list[TrainingExample],
        dataset: list[TrainingExample],
    ) -> float:
        # Parallel evaluation using task group
        scores: list[float] = []
        # Reuse semaphoe logic? Or let anyio manage tasks?
        # A semaphore would be good to limit concurrency here too.
        # Assuming we can define a semaphore for this instance or method.
        sem = anyio.Semaphore(10)  # Higher limit for evaluation?

        async def _eval_one(example: TrainingExample) -> float:
            async with sem:
                prompt = format_prompt(instruction, examples, example.inputs)
                try:
                    response = await self.llm_client.generate(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.config.target_model,
                        temperature=0.0,
                    )
                    return self.metric(response.content, example.reference)
                except BudgetExceededError:
                    raise
                except Exception as e:
                    logger.warning(f"Error during evaluation: {e}")
                    return 0.0

        async with anyio.create_task_group() as tg:
            for example in dataset:
                tg.start_soon(lambda ex: self._wrapper_eval(ex, scores, _eval_one), example)

        return sum(scores) / len(dataset) if dataset else 0.0

    async def _wrapper_eval(
        self, example: TrainingExample, score_list: list[float], func: Callable[[TrainingExample], Any]
    ) -> None:
        score = await func(example)
        score_list.append(score)

    async def compile(
        self,
        agent: Construct,
        trainset: list[TrainingExample],
        valset: list[TrainingExample],
    ) -> OptimizedManifest:
        logger.info(
            "Starting MIPRO compilation (Async)",
            train_size=len(trainset),
            target_model=self.config.target_model,
        )

        # 1. Diagnosis
        logger.info("Running baseline diagnosis...")
        dataset_obj = Dataset(trainset)
        failed_examples = []

        for example in trainset:
            prompt = format_prompt(agent.system_prompt, [], example.inputs)
            try:
                response = await self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config.target_model,
                    temperature=0.0,
                )
                score = self.metric(response.content, example.reference)
                if score < 1.0:
                    example.metadata["prediction"] = response.content
                    failed_examples.append(example)
            except BudgetExceededError:
                raise
            except Exception as e:
                logger.error(f"Error diagnosing example: {e}")

        logger.info(f"Diagnosis complete. Found {len(failed_examples)} failures.")

        # 2. Candidate Generation: Instructions
        instruction_candidates = {agent.system_prompt}
        logger.info(f"Generating {self.num_instruction_candidates} instruction candidates...")

        for i in range(self.num_instruction_candidates):
            try:
                new_instruction = await self.mutator.mutate(
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
        example_sets.append([])

        logger.info(f"Generating {self.num_fewshot_combinations} few-shot sets...")
        for _ in range(self.num_fewshot_combinations):
            k = self.config.max_bootstrapped_demos
            selected = await self.selector.select(dataset_obj, k=k)
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
                score = await self._evaluate_candidate(instr, ex_set, trainset)
                logger.debug(f"Candidate Score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_instruction = instr
                    best_examples = ex_set

        logger.info(f"Grid Search complete. Best Training Score: {best_score}")

        # 5. Final Evaluation
        final_metric = best_score
        if valset:
            logger.info("Evaluating best candidate on Validation Set...")
            final_metric = await self._evaluate_candidate(best_instruction, best_examples, valset)
            logger.info(f"Validation Score: {final_metric}")

        return OptimizedManifest(
            agent_id="unknown_agent",
            base_model=self.config.target_model,
            optimized_instruction=best_instruction,
            few_shot_examples=best_examples,
            performance_metric=final_metric,
            optimization_run_id=f"opt_mipro_{uuid.uuid4().hex[:8]}",
        )


class MiproOptimizer(PromptOptimizer):
    """
    Sync Facade for MiproOptimizerAsync.
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
        if hasattr(llm_client, "_async_client"):
            async_client = llm_client._async_client
        else:
            async_client = SyncToAsyncLLMClientAdapter(llm_client)

        async_embedding_provider = None
        if embedding_provider:
            if hasattr(embedding_provider, "_async_client"):
                async_embedding_provider = embedding_provider._async_client
            else:
                async_embedding_provider = SyncToAsyncEmbeddingProviderAdapter(embedding_provider)

        self._async = MiproOptimizerAsync(
            async_client,
            metric,
            config,
            async_embedding_provider,
            num_instruction_candidates,
            num_fewshot_combinations,
        )

    def compile(
        self,
        agent: Construct,
        trainset: list[TrainingExample],
        valset: list[TrainingExample],
    ) -> OptimizedManifest:
        try:

            async def _run() -> OptimizedManifest:
                return await self._async.compile(agent, trainset, valset)

            return cast(OptimizedManifest, anyio.run(_run))
        except Exception as e:
            raise unwrap_exception_group(e) from e
