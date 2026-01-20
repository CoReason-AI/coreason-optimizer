# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from coreason_optimizer.core.interfaces import UsageStats
from coreason_optimizer.utils.logger import logger


class BudgetExceededError(Exception):
    """Raised when the optimization budget is exceeded."""

    pass


class BudgetManager:
    """Tracks token usage and cost, enforcing a budget limit."""

    def __init__(self, budget_limit_usd: float) -> None:
        if budget_limit_usd <= 0:
            raise ValueError("Budget limit must be positive.")
        self.budget_limit_usd = budget_limit_usd
        self.total_cost_usd = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def consume(self, usage: UsageStats) -> None:
        """Accumulate usage stats."""
        self.total_cost_usd += usage.cost_usd
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

        logger.debug(
            f"Budget consumed: ${usage.cost_usd:.4f}. Total: ${self.total_cost_usd:.4f} / ${self.budget_limit_usd:.2f}"
        )

        self.check_budget()

    def check_budget(self) -> None:
        """Check if the budget has been exceeded."""
        if self.total_cost_usd > self.budget_limit_usd:
            msg = f"Budget exceeded! Spent ${self.total_cost_usd:.4f}, limit was ${self.budget_limit_usd:.2f}"
            logger.error(msg)
            raise BudgetExceededError(msg)

    def get_status(self) -> str:
        """Return a string summary of the budget status."""
        percentage = (self.total_cost_usd / self.budget_limit_usd) * 100 if self.budget_limit_usd > 0 else 100.0
        return f"Spent ${self.total_cost_usd:.4f} / ${self.budget_limit_usd:.2f} ({percentage:.1f}%)"
