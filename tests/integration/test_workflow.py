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

import pytest

from coreason_optimizer.core.models import OptimizedManifest, TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.mutator import IdentityMutator
from coreason_optimizer.strategies.selector import RandomSelector


@pytest.mark.asyncio
async def test_optimization_workflow_simulation() -> None:
    """Simulate a simple optimization loop (without the actual Loop class)."""

    # 1. Load Data
    raw_examples = [TrainingExample(inputs={"q": f"q{i}"}, reference=f"a{i}") for i in range(20)]
    dataset = Dataset(raw_examples)

    # 2. Split Data
    train_set, val_set, test_set = dataset.split(train_ratio=0.5, val_ratio=0.25)
    assert len(train_set) == 10
    assert len(val_set) == 5
    assert len(test_set) == 5

    # 3. Select Few-Shot Examples (Strategy)
    selector = RandomSelector(seed=999)
    few_shot_examples = await selector.select(train_set, k=3)
    assert len(few_shot_examples) == 3

    # 4. Mutate Instruction (Strategy)
    llm_mock = Mock()
    mutator = IdentityMutator(llm_client=llm_mock)
    base_instruction = "Answer the question."
    # Simulate finding failures in validation (mocked)
    failures = [TrainingExample(inputs={"q": "q_fail_1"}, reference="a_fail_1")]

    optimized_instruction = await mutator.mutate(current_instruction=base_instruction, failed_examples=failures)
    assert optimized_instruction == base_instruction  # Identity mutator

    # 5. Create Manifest (Artifact)
    # Assume we evaluated it and got a score
    score = 0.85

    manifest = OptimizedManifest(
        agent_id="test_agent_v1",
        base_model="gpt-4o",
        optimized_instruction=optimized_instruction,
        few_shot_examples=few_shot_examples,
        performance_metric=score,
        optimization_run_id="run_sim_001",
    )

    assert manifest.performance_metric == 0.85
    assert len(manifest.few_shot_examples) == 3
    assert manifest.optimized_instruction == "Answer the question."
