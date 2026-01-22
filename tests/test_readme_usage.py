# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_optimizer.core.client import OpenAIClientAsync
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import Construct
from coreason_optimizer.core.metrics import MetricFactory
from coreason_optimizer.core.models import OptimizedManifest
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.mipro import MiproOptimizer


class MockAgent:
    """A valid agent for testing."""

    system_prompt = "You are a helper."
    inputs = ["question"]
    outputs = ["answer"]


@pytest.mark.asyncio
async def test_readme_library_usage_flow(tmp_path: Path) -> None:
    """
    Verify the code snippet in 'Library Usage' section of README.md works.
    """
    # Setup dummy data
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("question,answer\nq1,a1\n", encoding="utf-8")

    val_csv = tmp_path / "val.csv"
    val_csv.write_text("question,answer\nq2,a2\n", encoding="utf-8")

    # 1. Configure
    config = OptimizerConfig(
        target_model="gpt-4o",
        budget_limit_usd=5.0,
        max_rounds=1,  # Reduced for test speed
    )

    # 2. Initialize Components
    # Mock AsyncOpenAI to avoid API calls and key errors
    with patch("coreason_optimizer.core.client.AsyncOpenAI") as MockOpenAI:
        mock_client_instance = AsyncMock()
        MockOpenAI.return_value = mock_client_instance

        # Mock generate response for Mipro diagnosis/eval
        # The MiproOptimizer calls client.generate (which we wrap)
        # But if we use OpenAIClientAsync, it calls client.chat.completions.create

        # We need to mock client.chat.completions.create
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "a1"
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 10
        mock_completion.usage.total_tokens = 20

        mock_client_instance.chat.completions.create.return_value = mock_completion

        # Async client usage
        async with OpenAIClientAsync() as client:
            metric = MetricFactory.get("exact_match")

            optimizer = MiproOptimizer(client, metric, config)

            # 3. Load Data
            train_set = Dataset.from_csv(train_csv, input_cols=["question"], reference_col="answer")
            val_set = Dataset.from_csv(val_csv, input_cols=["question"], reference_col="answer")

            # 4. Compile
            agent = MockAgent()
            manifest = await optimizer.compile(agent, list(train_set), list(val_set))

            # 5. Save (Verify object)
            assert isinstance(manifest, OptimizedManifest)
            assert manifest.base_model == "gpt-4o"
            assert manifest.performance_metric is not None


@pytest.mark.asyncio
async def test_invalid_agent_protocol() -> None:
    """Test using an agent that does not satisfy the Construct protocol."""

    class InvalidAgent:
        # Missing inputs/outputs
        system_prompt = "broken"

    config = OptimizerConfig()
    with patch("coreason_optimizer.core.client.AsyncOpenAI"):
        async with OpenAIClientAsync() as client:
            metric = MetricFactory.get("exact_match")
            optimizer = MiproOptimizer(client, metric, config)

            # MiproOptimizer.compile calls methods on agent.
            # It handles missing attributes gracefully or crashes depending on usage.
            # Since we use mocks and Empty dataset (implicit in diagnosis if we don't pass one?
            # No, diagnosis iterates trainset). We need to pass a trainset to trigger agent usage.

            # Cast to Construct to bypass mypy check since we are testing runtime behavior
            agent = cast(Construct, InvalidAgent())
            manifest = await optimizer.compile(agent, [], [])
            assert isinstance(manifest, OptimizedManifest)


def test_unknown_metric() -> None:
    """Test requesting an unknown metric."""
    with pytest.raises(ValueError, match="Unknown metric: unknown"):
        MetricFactory.get("unknown")


@pytest.mark.asyncio
async def test_mipro_missing_embeddings_for_semantic() -> None:
    """Test error when semantic selector is requested but no embedding provider is given."""
    config = OptimizerConfig(selector_type="semantic")

    with patch("coreason_optimizer.core.client.AsyncOpenAI"):
        async with OpenAIClientAsync() as client:
            metric = MetricFactory.get("exact_match")

            # Should raise ValueError in __init__
            with pytest.raises(ValueError, match="Embedding provider is required for semantic selection"):
                MiproOptimizer(client, metric, config, embedding_provider=None)
