# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

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
    """
    if ":" in agent_path_str:
        file_path_str, variable_name = agent_path_str.split(":", 1)
    else:
        file_path_str = agent_path_str
        variable_name = "agent"

    path = Path(file_path_str)
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")

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
