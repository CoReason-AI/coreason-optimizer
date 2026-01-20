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

from coreason_optimizer.core.budget import BudgetExceededError, BudgetManager
from coreason_optimizer.core.interfaces import UsageStats


def test_zero_budget_limit_init() -> None:
    """Test that initializing with 0 budget is rejected."""
    with pytest.raises(ValueError, match="Budget limit must be positive"):
        BudgetManager(0.0)


def test_exact_budget_limit() -> None:
    """Test that hitting the exact budget limit does NOT raise an error."""
    bm = BudgetManager(10.0)
    usage = UsageStats(cost_usd=10.0)

    bm.consume(usage)  # Should not raise
    assert bm.total_cost_usd == 10.0


def test_slightly_over_budget() -> None:
    """Test that exceeding budget by a tiny amount raises error."""
    bm = BudgetManager(10.0)
    # Using epsilon
    usage = UsageStats(cost_usd=10.0000001)

    with pytest.raises(BudgetExceededError):
        bm.consume(usage)


def test_negative_usage_rejection() -> None:
    """Test that negative usage stats are rejected."""
    bm = BudgetManager(10.0)

    with pytest.raises(ValueError, match="Usage stats cannot be negative"):
        bm.consume(UsageStats(cost_usd=-1.0))

    with pytest.raises(ValueError, match="Usage stats cannot be negative"):
        bm.consume(UsageStats(total_tokens=-10))


def test_complex_consumption_loop() -> None:
    """Simulate a loop of transactions until failure."""
    budget_limit = 5.0
    bm = BudgetManager(budget_limit)
    cost_per_step = 0.3

    steps_taken = 0
    total_spent = 0.0

    # We expect to run until total_spent > 5.0
    # 5.0 / 0.3 = 16.666... So 16 steps = 4.8. 17 steps = 5.1 (Fail)

    try:
        for _ in range(100):
            bm.consume(UsageStats(cost_usd=cost_per_step))
            # If consume raises, we don't reach here
            steps_taken += 1
            total_spent += cost_per_step
    except BudgetExceededError:
        pass

    # The 17th step triggers the error inside consume().
    # Step 1: 0.3. steps_taken=1.
    # ...
    # Step 16: 4.8. steps_taken=16.
    # Step 17: 5.1. Raises. steps_taken remains 16.
    assert steps_taken == 16

    assert bm.total_cost_usd > budget_limit
    assert abs(bm.total_cost_usd - 5.1) < 1e-9
