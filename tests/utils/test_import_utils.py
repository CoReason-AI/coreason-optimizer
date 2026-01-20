# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from pathlib import Path

import pytest

from coreason_optimizer.core.interfaces import Construct
from coreason_optimizer.utils.import_utils import load_agent_from_path


# Mock Construct implementation
class ValidAgent:
    def __init__(self) -> None:
        self.system_prompt = "You are a helpful assistant."
        self.inputs = ["question"]
        self.outputs = ["answer"]


class InvalidAgent:
    # Missing system_prompt
    def __init__(self) -> None:
        self.inputs = ["q"]
        self.outputs = ["a"]


def test_load_agent_success_default_var(tmp_path: Path) -> None:
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "my_agent.py"
    content = """
class ValidAgent:
    @property
    def system_prompt(self): return "prompt"
    @property
    def inputs(self): return ["i"]
    @property
    def outputs(self): return ["o"]

agent = ValidAgent()
    """
    p.write_text(content, encoding="utf-8")

    loaded = load_agent_from_path(str(p))
    assert isinstance(loaded, Construct)
    assert loaded.system_prompt == "prompt"


def test_load_agent_success_custom_var(tmp_path: Path) -> None:
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "custom.py"
    content = """
class ValidAgent:
    @property
    def system_prompt(self): return "prompt_custom"
    @property
    def inputs(self): return ["i"]
    @property
    def outputs(self): return ["o"]

my_custom_agent = ValidAgent()
    """
    p.write_text(content, encoding="utf-8")

    path_str = f"{p}:my_custom_agent"
    loaded = load_agent_from_path(path_str)
    assert loaded.system_prompt == "prompt_custom"


def test_load_agent_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_agent_from_path("non_existent_file.py")


def test_load_agent_var_not_found(tmp_path: Path) -> None:
    p = tmp_path / "empty.py"
    p.write_text("x = 1", encoding="utf-8")
    with pytest.raises(AttributeError):
        load_agent_from_path(str(p))


def test_load_agent_invalid_protocol(tmp_path: Path) -> None:
    p = tmp_path / "bad.py"
    content = """
class BadAgent:
    pass
agent = BadAgent()
    """
    p.write_text(content, encoding="utf-8")
    with pytest.raises(TypeError, match="Missing attributes"):
        load_agent_from_path(str(p))


def test_load_agent_import_error(tmp_path: Path) -> None:
    p = tmp_path / "syntax_error.py"
    p.write_text("this is not python", encoding="utf-8")
    with pytest.raises(ImportError):
        load_agent_from_path(str(p))
