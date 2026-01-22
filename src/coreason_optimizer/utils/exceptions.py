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
Custom exceptions for the Coreason Optimizer.
"""


class OptimizerError(Exception):
    """Base class for all optimizer exceptions."""

    pass


class BudgetExceededError(OptimizerError):
    """Raised when the optimization budget (USD) is exceeded."""

    pass


class ConfigurationError(OptimizerError):
    """Raised when the configuration is invalid."""

    pass
