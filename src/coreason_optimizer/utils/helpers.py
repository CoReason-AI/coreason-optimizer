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
Helper utilities.
"""

from typing import TypeVar

from coreason_optimizer.core.budget import BudgetExceededError

T = TypeVar("T")


def unwrap_exception_group(exc: Exception) -> Exception:
    """
    Unwraps an ExceptionGroup.
    Prioritizes BudgetExceededError if present.
    If multiple exceptions, returns the first one.
    """
    if isinstance(exc, BaseExceptionGroup):
        # Check if BudgetExceededError is in there
        for e in exc.exceptions:
            if isinstance(e, BudgetExceededError):
                return e
            # If nested group, recurse?
            if isinstance(e, BaseExceptionGroup):
                unwrapped = unwrap_exception_group(e)
                if isinstance(unwrapped, BudgetExceededError):
                    return unwrapped

        # Return first exception if no priority one found
        if exc.exceptions:
            return exc.exceptions[0]  # type: ignore
    return exc
