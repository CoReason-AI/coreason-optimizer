# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import sys
from unittest.mock import MagicMock, patch

import pytest

from coreason_optimizer.core.interfaces import Construct
from coreason_optimizer.utils.import_utils import load_agent_from_path


# Mock Construct
class MockAgent(Construct):
    system_prompt: str = "System"
    inputs: list[str] = ["input"]
    outputs: list[str] = ["output"]


def test_load_agent_from_path_success(tmp_path):
    """Test successful loading of an agent from a file."""
    # Create a temporary python file
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "my_agent.py"
    p.touch()  # Create the file so existence check passes

    with patch("importlib.util.spec_from_file_location") as mock_spec_from_file:
        mock_spec = MagicMock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = MagicMock()
        # Set the agent attribute
        mock_module.agent = MockAgent()

        with patch("importlib.util.module_from_spec", return_value=mock_module) as mock_module_from_spec:
            with patch.object(sys, "modules", {}):  # Isolate sys.modules
                agent = load_agent_from_path(str(p))
                assert isinstance(agent, MockAgent)
                assert agent.system_prompt == "System"


def test_load_agent_from_path_with_variable(tmp_path):
    """Test loading an agent with a specific variable name."""
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "my_agent.py"
    p.touch()
    path_str = f"{p}:my_custom_agent"

    with patch("importlib.util.spec_from_file_location") as mock_spec_from_file:
        mock_spec = MagicMock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = MagicMock()
        mock_module.my_custom_agent = MockAgent()

        with patch("importlib.util.module_from_spec", return_value=mock_module):
            agent = load_agent_from_path(path_str)
            assert isinstance(agent, MockAgent)


def test_load_agent_file_not_found():
    """Test FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_agent_from_path("non_existent_file.py")


def test_load_agent_module_spec_failure(tmp_path):
    """Test ImportError when spec cannot be loaded."""
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "bad_agent.py"
    p.touch()

    with patch("importlib.util.spec_from_file_location", return_value=None):
        with pytest.raises(ImportError, match="Could not load spec"):
            load_agent_from_path(str(p))


def test_load_agent_execution_failure(tmp_path):
    """Test ImportError when module execution fails."""
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "error_agent.py"
    p.touch()

    with patch("importlib.util.spec_from_file_location") as mock_spec_from_file:
        mock_spec = MagicMock()
        mock_spec_from_file.return_value = mock_spec
        mock_spec.loader.exec_module.side_effect = Exception("Module Error")

        with patch("importlib.util.module_from_spec", return_value=MagicMock()):
            with pytest.raises(ImportError, match="Error executing module"):
                load_agent_from_path(str(p))


def test_load_agent_variable_not_found(tmp_path):
    """Test AttributeError when variable is missing."""
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "no_var_agent.py"
    p.touch()

    with patch("importlib.util.spec_from_file_location") as mock_spec_from_file:
        mock_spec = MagicMock()
        mock_spec_from_file.return_value = mock_spec

        # When module_from_spec returns a mock, hasattr(mock, "agent") is usually True (it creates a new mock).
        # We need to configure the mock to NOT have 'agent'.
        mock_module = MagicMock()
        del mock_module.agent  # Explicitly delete it if it auto-created, or use spec

        # Alternative: use a real class instance
        class EmptyModule:
            pass

        with patch("importlib.util.module_from_spec", return_value=EmptyModule()):
            with pytest.raises(AttributeError, match="Variable 'agent' not found"):
                load_agent_from_path(str(p))


def test_load_agent_protocol_mismatch(tmp_path):
    """Test TypeError when object does not satisfy protocol."""
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "bad_protocol.py"
    p.touch()

    class BadAgent:
        pass  # Missing attributes

    with patch("importlib.util.spec_from_file_location") as mock_spec_from_file:
        mock_spec = MagicMock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = MagicMock()
        mock_module.agent = BadAgent()

        with patch("importlib.util.module_from_spec", return_value=mock_module):
            with pytest.raises(TypeError, match="does not satisfy Construct protocol"):
                load_agent_from_path(str(p))
