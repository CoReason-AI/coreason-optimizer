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
from typing import Any, Callable, cast

import anyio

from coreason_optimizer.core.async_client import BudgetAwareLLMClientAsync
from coreason_optimizer.core.budget import BudgetExceededError, BudgetManager
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.formatter import format_prompt
from coreason_optimizer.core.interfaces import (
    Construct,
    LLMClient,
    LLMClientAsync,
    Metric,
    PromptOptimizer,
    PromptOptimizerAsync,
)
from coreason_optimizer.core.models import OptimizedManifest, TrainingExample
from coreason_optimizer.utils.adapters import SyncToAsyncLLMClientAdapter
from coreason_optimizer.utils.helpers import unwrap_exception_group
from coreason_optimizer.utils.logger import logger


class BootstrapFewShotAsync(PromptOptimizerAsync):
    """
    Async BootstrapFewShot strategy implementation.
    """

    def __init__(
        self,
        llm_client: LLMClientAsync,
        metric: Metric,
        config: OptimizerConfig,
    ):
        self.metric = metric
        self.config = config
        self.budget_manager = BudgetManager(config.budget_limit_usd)
        self.llm_client = BudgetAwareLLMClientAsync(llm_client, self.budget_manager)

    async def compile(
        self,
        agent: Construct,
        trainset: list[TrainingExample],
        valset: list[TrainingExample],
    ) -> OptimizedManifest:
        logger.info(
            "Starting BootstrapFewShot compilation (Async)",
            train_size=len(trainset),
            target_model=self.config.target_model,
        )

        successful_traces: list[TrainingExample] = []

        # 1. Mine successful traces
        # Use semaphore for rate limiting (e.g. 5 concurrent requests)
        sem = anyio.Semaphore(5)

        async def _process_example(example: TrainingExample, idx: int) -> TrainingExample | None:
            async with sem:
                prompt = format_prompt(
                    system_prompt=agent.system_prompt,
                    examples=[],
                    inputs=example.inputs,
                )

                try:
                    response = await self.llm_client.generate(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.config.target_model,
                        temperature=0.0,
                    )
                except BudgetExceededError:
                    raise
                except Exception as e:
                    logger.error(f"Error generating for example {idx}: {e}")
                    return None

                prediction = response.content
                score = self.metric(prediction, example.reference)

                if score >= 1.0:
                    logger.debug(f"Example {idx} passed with score {score}")
                    return example
                else:
                    logger.debug(f"Example {idx} failed with score {score}")
                    return None

        async with anyio.create_task_group() as tg:
            for i, example in enumerate(trainset):
                tg.start_soon(
                    lambda ex, idx: self._wrapper_process(ex, idx, successful_traces, _process_example), example, i
                )

        # 5. Select Candidates
        num_demos = min(len(successful_traces), self.config.max_bootstrapped_demos)
        selected_examples = successful_traces[:num_demos]

        logger.info(f"Bootstrapping complete. Selected {len(selected_examples)} examples.")

        # 6. Evaluate on Validation Set

        async def _evaluate_one(example: TrainingExample) -> float:
            async with sem:
                prompt = format_prompt(
                    system_prompt=agent.system_prompt,
                    examples=selected_examples,
                    inputs=example.inputs,
                )
                try:
                    response = await self.llm_client.generate(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.config.target_model,
                        temperature=0.0,
                    )
                    return self.metric(response.content, example.reference)
                except BudgetExceededError:
                    raise
                except Exception:
                    # Swallow other exceptions (e.g. transient network err) but NOT budget error
                    return 0.0

        total_score = 0.0
        if valset:
            scores: list[float] = []
            async with anyio.create_task_group() as tg:
                for example in valset:
                    tg.start_soon(lambda ex: self._wrapper_evaluate(ex, scores, _evaluate_one), example)

            total_score = sum(scores)
            avg_score = total_score / len(valset)
        else:
            avg_score = 0.0

        return OptimizedManifest(
            agent_id="unknown_agent",
            base_model=self.config.target_model,
            optimized_instruction=agent.system_prompt,
            few_shot_examples=selected_examples,
            performance_metric=avg_score,
            optimization_run_id=f"opt_{uuid.uuid4().hex[:8]}",
        )

    # Helper wrapper for task group
    async def _wrapper_process(
        self,
        example: TrainingExample,
        idx: int,
        result_list: list[TrainingExample],
        func: Callable[[TrainingExample, int], Any],
    ) -> None:
        res = await func(example, idx)
        if res:
            result_list.append(res)

    async def _wrapper_evaluate(
        self, example: TrainingExample, score_list: list[float], func: Callable[[TrainingExample], Any]
    ) -> None:
        score = await func(example)
        score_list.append(score)


class BootstrapFewShot(PromptOptimizer):
    """
    Sync Facade for BootstrapFewShotAsync.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        metric: Metric,
        config: OptimizerConfig,
    ):
        if hasattr(llm_client, "_async_client"):
            async_client = llm_client._async_client
        else:
            # Backward compatibility via adapter
            async_client = SyncToAsyncLLMClientAdapter(llm_client)

        self._async = BootstrapFewShotAsync(async_client, metric, config)

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
