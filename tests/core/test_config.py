# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import pytest
from coreason_optimizer.core.config import OptimizerConfig
from pydantic import ValidationError


def test_default_config() -> None:
    """Test that default values are set correctly."""
    config = OptimizerConfig()
    assert config.target_model == "gpt-4o"
    assert config.metric == "exact_match"
    assert config.max_bootstrapped_demos == 4
    assert config.max_rounds == 10
    assert config.budget_limit_usd == 10.0


def test_custom_config() -> None:
    """Test that custom values are set correctly."""
    config = OptimizerConfig(
        target_model="claude-3-opus",
        metric="f1_score",
        max_bootstrapped_demos=2,
        max_rounds=5,
        budget_limit_usd=50.0,
    )
    assert config.target_model == "claude-3-opus"
    assert config.metric == "f1_score"
    assert config.max_bootstrapped_demos == 2
    assert config.max_rounds == 5
    assert config.budget_limit_usd == 50.0


def test_validation_constraints() -> None:
    """Test that validation constraints are enforced."""
    with pytest.raises(ValidationError):
        # max_rounds must be > 0
        OptimizerConfig(max_rounds=0)

    with pytest.raises(ValidationError):
        # max_bootstrapped_demos must be >= 0
        OptimizerConfig(max_bootstrapped_demos=-1)

    with pytest.raises(ValidationError):
        # budget_limit_usd must be > 0
        OptimizerConfig(budget_limit_usd=0.0)
