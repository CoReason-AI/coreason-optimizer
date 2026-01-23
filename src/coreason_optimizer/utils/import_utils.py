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
Dynamic import utilities.

This module provides functionality to load python objects (specifically Agents)
from file paths, handling different path formats and protocol validation.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

from coreason_optimizer.core.interfaces import Construct


def load_agent_from_path(agent_path_str: str) -> Construct:
    """
    Load an agent (Construct) from a file path string.

    Format: "path/to/file.py" (defaults to variable 'agent')
            "path/to/file.py:variable_name"

    Args:
        agent_path_str: The path string to the python file and optional variable.

    Returns:
        The loaded agent object which conforms to the Construct protocol.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If the module cannot be imported.
        AttributeError: If the specified variable is not found in the module.
        TypeError: If the loaded object does not satisfy the Construct protocol.
    """
    file_path: Path | None = None
    variable_name = "agent"

    # 1. Try treating the whole string as a path
    p = Path(agent_path_str)
    if p.exists() and p.is_file():
        file_path = p
    else:
        # 2. Try splitting at the last colon (for path:variable)
        # Note: We use rsplit to handle Windows drive letters (C:\...) correctly
        # because a drive letter colon is near the start, and separator is near the end.
        if ":" in agent_path_str:
            parts = agent_path_str.rsplit(":", 1)
            # If split results in 2 parts, check if the first part is a valid file
            if len(parts) == 2:
                possible_path = Path(parts[0])
                if possible_path.exists() and possible_path.is_file():
                    file_path = possible_path
                    variable_name = parts[1]

    if file_path is None:
        raise FileNotFoundError(f"Agent file not found: {agent_path_str}")

    path = file_path
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for file: {path}")  # pragma: no cover

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {path}: {e}") from e

    if not hasattr(module, variable_name):
        raise AttributeError(f"Variable '{variable_name}' not found in {path}")

    agent_obj: Any = getattr(module, variable_name)

    # Basic Protocol check (runtime)
    # Since Construct is @runtime_checkable, isinstance works for properties if implemented as properties.
    # However, Protocols with properties are tricky with isinstance check on instances that
    # implement them as instance vars. We will do a manual check for safety.
    if not isinstance(agent_obj, Construct):
        # Double check: maybe it has the attributes but isinstance failed due to some typing quirk?
        # Let's check explicitly.
        required_attrs = ["inputs", "outputs", "system_prompt"]
        missing = [attr for attr in required_attrs if not hasattr(agent_obj, attr)]
        if missing:
            raise TypeError(
                f"Agent object '{variable_name}' does not satisfy Construct protocol. Missing attributes: {missing}"
            )

    return cast(Construct, agent_obj)
