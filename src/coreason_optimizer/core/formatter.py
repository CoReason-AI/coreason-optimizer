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
Prompt formatting utilities.

This module provides functions to construct the final prompt string from
system instructions, few-shot examples, and user inputs.
"""

from typing import Any

from coreason_optimizer.core.models import TrainingExample


def format_prompt(
    system_prompt: str,
    examples: list[TrainingExample],
    inputs: dict[str, Any],
) -> str:
    """
    Format a complete prompt with system instruction, few-shot examples, and user input.

    Format Structure:
    ### System Instruction
    {system_prompt}

    ### Examples
    Input: {example_input}
    Output: {example_output}
    ...

    ### User Input
    Input: {inputs}

    Args:
        system_prompt: The system instruction text.
        examples: A list of TrainingExample objects to use as few-shot demonstrations.
        inputs: A dictionary of input variables for the current query.

    Returns:
        The formatted prompt string.
    """
    parts = []

    # System Prompt
    parts.append(f"### System Instruction\n{system_prompt}")

    # Examples
    if examples:
        parts.append("### Examples")
        for ex in examples:
            # We assume inputs are dicts, we serialize them simply
            input_str = ", ".join(f"{k}: {v}" for k, v in ex.inputs.items())
            parts.append(f"Input: {input_str}\nOutput: {ex.reference}")

    # User Input
    parts.append("### User Input")
    current_input_str = ", ".join(f"{k}: {v}" for k, v in inputs.items())
    parts.append(f"Input: {current_input_str}")

    return "\n\n".join(parts)
