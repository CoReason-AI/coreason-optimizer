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


def test_budget_initialization() -> None:
    bm = BudgetManager(10.0)
    assert bm.budget_limit_usd == 10.0
    assert bm.total_cost_usd == 0.0
    assert bm.total_tokens == 0

    with pytest.raises(ValueError):
        BudgetManager(-1.0)


def test_budget_consumption() -> None:
    bm = BudgetManager(10.0)
    usage = UsageStats(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost_usd=0.5,
    )

    bm.consume(usage)
    assert bm.total_cost_usd == 0.5
    assert bm.total_tokens == 30
    assert bm.total_prompt_tokens == 10

    bm.consume(usage)
    assert bm.total_cost_usd == 1.0
    assert bm.total_tokens == 60


def test_budget_exceeded() -> None:
    bm = BudgetManager(1.0)
    usage = UsageStats(cost_usd=0.6)

    bm.consume(usage)  # 0.6 <= 1.0, OK

    with pytest.raises(BudgetExceededError) as exc:
        bm.consume(usage)  # 1.2 > 1.0, Fail

    assert "Budget exceeded" in str(exc.value)
    assert "Spent $1.2000" in str(exc.value)


def test_budget_status_formatting() -> None:
    bm = BudgetManager(100.0)
    bm.consume(UsageStats(cost_usd=50.0))
    status = bm.get_status()
    assert "Spent $50.0000 / $100.00" in status
    assert "(50.0%)" in status
