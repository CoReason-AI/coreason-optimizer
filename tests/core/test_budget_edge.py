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

from coreason_optimizer.core.budget import BudgetManager
from coreason_optimizer.core.interfaces import UsageStats


def test_budget_init_negative() -> None:
    with pytest.raises(ValueError):
        BudgetManager(-1.0)
    with pytest.raises(ValueError):
        BudgetManager(0.0)


def test_budget_consume_negative() -> None:
    manager = BudgetManager(10.0)
    stats = UsageStats(total_tokens=10, prompt_tokens=5, completion_tokens=5, cost_usd=-0.1)
    with pytest.raises(ValueError):
        manager.consume(stats)

    stats_bad_tokens = UsageStats(total_tokens=-1, prompt_tokens=5, completion_tokens=5, cost_usd=0.1)
    with pytest.raises(ValueError):
        manager.consume(stats_bad_tokens)


def test_budget_status_string() -> None:
    manager = BudgetManager(10.0)
    assert manager.get_status() == "Spent $0.0000 / $10.00 (0.0%)"

    manager.consume(UsageStats(total_tokens=10, prompt_tokens=5, completion_tokens=5, cost_usd=5.0))
    assert "50.0%" in manager.get_status()
