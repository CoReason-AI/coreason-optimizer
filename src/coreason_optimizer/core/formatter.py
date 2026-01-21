# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from typing import Any

from coreason_optimizer.core.models import TrainingExample


def format_prompt(
    system_prompt: str,
    examples: list[TrainingExample],
    inputs: dict[str, Any],
) -> str:
    """
    Sensible default prompt formatter.

    Structure:
    ### System Instruction
    ...
    ### Examples
    Input: ...
    Output: ...
    ### User Input
    Input: ...
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
