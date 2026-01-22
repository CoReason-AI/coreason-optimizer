# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from coreason_optimizer.core.models import OptimizedManifest, TrainingExample
from coreason_optimizer.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_agent_file(tmp_path: MagicMock) -> str:
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "test_agent.py"
    p.touch()
    return str(p)


@pytest.fixture
def mock_dataset_file(tmp_path: MagicMock) -> str:
    d = tmp_path / "data"
    d.mkdir()
    p = d / "data.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"input": "q1", "reference": "a1"}) + "\n")
    return str(p)


@pytest.fixture
def mock_manifest_file(tmp_path: MagicMock) -> str:
    d = tmp_path / "out"
    d.mkdir()
    p = d / "manifest.json"
    manifest = OptimizedManifest(
        agent_id="agent",
        base_model="gpt-4",
        optimized_instruction="Instr",
        few_shot_examples=[TrainingExample(inputs={"q": "ex"}, reference="ref")],
        performance_metric=1.0,
        optimization_run_id="run",
    )
    with open(p, "w") as f:
        f.write(manifest.model_dump_json())
    return str(p)


def test_tune_metric_error(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    """Test ValueError when getting metric in tune command."""
    with patch("coreason_optimizer.main.load_agent_from_path"):
        with patch("coreason_optimizer.main.OpenAIClient"):
            with patch("coreason_optimizer.main.MetricFactory.get", side_effect=ValueError("Invalid Metric")):
                # Removed --metric option as it doesn't exist in tune command
                result = runner.invoke(cli, ["tune", "--agent", mock_agent_file, "--dataset", mock_dataset_file])
                assert result.exit_code != 0
                assert "Invalid Metric" in result.output


def test_tune_embedding_init_error(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    """Test exception when initializing embedding client in tune command."""
    with patch("coreason_optimizer.main.load_agent_from_path"):
        with patch("coreason_optimizer.main.OpenAIClient"):
            with patch("coreason_optimizer.main.OpenAIEmbeddingClient", side_effect=Exception("Embed Init Fail")):
                result = runner.invoke(
                    cli, ["tune", "--agent", mock_agent_file, "--dataset", mock_dataset_file, "--selector", "semantic"]
                )
                assert result.exit_code != 0
                assert "Failed to initialize OpenAI Embedding Client" in result.output


def test_tune_compile_exception(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    """Test generic exception during optimization compilation."""
    with patch("coreason_optimizer.main.load_agent_from_path"):
        with patch("coreason_optimizer.main.OpenAIClient"):
            # Mock MiproOptimizer.compile to raise
            with patch(
                "coreason_optimizer.strategies.mipro.MiproOptimizer.compile", side_effect=Exception("Compile Error")
            ):
                result = runner.invoke(cli, ["tune", "--agent", mock_agent_file, "--dataset", mock_dataset_file])
                assert result.exit_code != 0
                assert "Optimization failed: Compile Error" in result.output


def test_tune_save_exception(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    """Test exception when saving the manifest."""
    with patch("coreason_optimizer.main.load_agent_from_path"):
        with patch("coreason_optimizer.main.OpenAIClient"):
            with patch("coreason_optimizer.strategies.bootstrap.BootstrapFewShot.compile") as mock_compile:
                mock_compile.return_value = OptimizedManifest(
                    agent_id="test",
                    base_model="m",
                    optimized_instruction="i",
                    few_shot_examples=[],
                    performance_metric=1,
                    optimization_run_id="1",
                )
                # Output to a directory which causes IsADirectoryError (mapped to Exception)
                result = runner.invoke(
                    cli,
                    [
                        "tune",
                        "--agent",
                        mock_agent_file,
                        "--dataset",
                        mock_dataset_file,
                        "--strategy",
                        "bootstrap",
                        "--output",
                        "/",
                    ],
                )
                assert result.exit_code != 0
                # The message format depends on OS, but click exception wraps it
                assert (
                    "Is a directory" in result.output
                    or "Permission denied" in result.output
                    or "Failed to save manifest" in result.output
                )


def test_evaluate_client_init_exception(runner: CliRunner, mock_manifest_file: str, mock_dataset_file: str) -> None:
    """Test exception when initializing OpenAI client in evaluate command."""
    with patch("coreason_optimizer.main.OpenAIClient", side_effect=Exception("Client Init Fail")):
        result = runner.invoke(cli, ["evaluate", "--manifest", mock_manifest_file, "--dataset", mock_dataset_file])
        assert result.exit_code != 0
        assert "Failed to initialize OpenAI Client" in result.output


def test_evaluate_metric_exception(runner: CliRunner, mock_manifest_file: str, mock_dataset_file: str) -> None:
    """Test exception when getting metric in evaluate command."""
    with patch("coreason_optimizer.main.OpenAIClient"):
        with patch("coreason_optimizer.main.MetricFactory.get", side_effect=ValueError("Bad Metric")):
            result = runner.invoke(
                cli, ["evaluate", "--manifest", mock_manifest_file, "--dataset", mock_dataset_file, "--metric", "bad"]
            )
            assert result.exit_code != 0
            assert "Bad Metric" in result.output


def test_evaluate_manifest_load_error(runner: CliRunner, mock_dataset_file: str) -> None:
    """Test exception when loading manifest fails."""
    # Pass a non-existent file
    result = runner.invoke(cli, ["evaluate", "--manifest", "nonexistent.json", "--dataset", mock_dataset_file])
    assert result.exit_code != 0
    assert "Failed to load manifest" in result.output


def test_evaluate_dataset_load_error(runner: CliRunner, mock_manifest_file: str) -> None:
    """Test exception when loading dataset fails."""
    result = runner.invoke(cli, ["evaluate", "--manifest", mock_manifest_file, "--dataset", "nonexistent.jsonl"])
    assert result.exit_code != 0
    assert "Failed to load dataset" in result.output
