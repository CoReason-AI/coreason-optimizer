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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from coreason_optimizer.core.models import OptimizedManifest, TrainingExample
from coreason_optimizer.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_agent_file(tmp_path: Path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    p = d / "test_agent.py"
    content = """
class TestAgent:
    @property
    def system_prompt(self): return "sys"
    @property
    def inputs(self): return ["q"]
    @property
    def outputs(self): return ["a"]
agent = TestAgent()
"""
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def mock_dataset_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.jsonl"
    data = [
        {"input": {"q": "hi"}, "output": "hello"},
        {"input": {"q": "bye"}, "output": "goodbye"},
    ]
    with open(p, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return p


@pytest.fixture
def mock_manifest_file(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.json"
    manifest = OptimizedManifest(
        agent_id="test",
        base_model="gpt-4o",
        optimized_instruction="opt_sys",
        few_shot_examples=[TrainingExample(inputs={"q": "ex_q"}, reference="ex_a")],
        performance_metric=1.0,
        optimization_run_id="123",
    )
    p.write_text(manifest.model_dump_json(), encoding="utf-8")
    return p


def test_tune_command_success(
    runner: CliRunner, mock_agent_file: Path, mock_dataset_file: Path, tmp_path: Path
) -> None:
    output_path = tmp_path / "out.json"

    with patch("coreason_optimizer.main.OpenAIClient"), patch("coreason_optimizer.main.MiproOptimizer") as MockMipro:
        # Setup mock return for compile
        mock_instance = MockMipro.return_value
        mock_instance.compile.return_value = OptimizedManifest(
            agent_id="test",
            base_model="gpt-4o",
            optimized_instruction="new_sys",
            few_shot_examples=[],
            performance_metric=0.9,
            optimization_run_id="run_1",
        )

        result = runner.invoke(
            cli,
            [
                "tune",
                "--agent",
                str(mock_agent_file),
                "--dataset",
                str(mock_dataset_file),
                "--output",
                str(output_path),
                "--base-model",
                "gpt-3.5-turbo",
                "--epochs",
                "5",
                "--demos",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert "Optimization complete" in result.output
        assert output_path.exists()

        # Verify args passed
        assert MockMipro.called
        # Check config arg
        args, _ = MockMipro.call_args
        config = args[2]
        assert config.target_model == "gpt-3.5-turbo"
        assert config.max_rounds == 5
        assert config.max_bootstrapped_demos == 2


def test_tune_command_bootstrap(
    runner: CliRunner, mock_agent_file: Path, mock_dataset_file: Path, tmp_path: Path
) -> None:
    output = tmp_path / "bootstrap.json"
    with (
        patch("coreason_optimizer.main.OpenAIClient"),
        patch("coreason_optimizer.main.BootstrapFewShot") as MockBootstrap,
    ):
        mock_instance = MockBootstrap.return_value
        mock_instance.compile.return_value = OptimizedManifest(
            agent_id="test",
            base_model="gpt-4o",
            optimized_instruction="sys",
            few_shot_examples=[],
            performance_metric=0.8,
            optimization_run_id="run_2",
        )

        result = runner.invoke(
            cli,
            [
                "tune",
                "--agent",
                str(mock_agent_file),
                "--dataset",
                str(mock_dataset_file),
                "--strategy",
                "bootstrap",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert MockBootstrap.called
        assert output.exists()


def test_tune_agent_not_found(runner: CliRunner, mock_dataset_file: Path) -> None:
    result = runner.invoke(cli, ["tune", "--agent", "missing.py", "--dataset", str(mock_dataset_file)])
    assert result.exit_code != 0
    assert "Failed to load agent" in result.output or isinstance(result.exception, SystemExit)


def test_tune_dataset_not_found(runner: CliRunner, mock_agent_file: Path) -> None:
    result = runner.invoke(cli, ["tune", "--agent", str(mock_agent_file), "--dataset", "missing.csv"])
    assert result.exit_code != 0
    # ClickException just prints the message
    assert "File not found" in result.output


def test_tune_openai_client_fail(runner: CliRunner, mock_agent_file: Path, mock_dataset_file: Path) -> None:
    with patch("coreason_optimizer.main.OpenAIClient", side_effect=Exception("API Key missing")):
        result = runner.invoke(
            cli,
            ["tune", "--agent", str(mock_agent_file), "--dataset", str(mock_dataset_file)],
        )
        assert result.exit_code != 0
        assert "Failed to initialize OpenAI Client" in result.output


def test_tune_metric_fail(runner: CliRunner, mock_agent_file: Path, mock_dataset_file: Path) -> None:
    # Need to mock Config default metric or override it
    with (
        patch("coreason_optimizer.main.OpenAIClient"),
        patch("coreason_optimizer.main.MetricFactory.get", side_effect=ValueError("Unknown metric")),
    ):
        result = runner.invoke(
            cli,
            ["tune", "--agent", str(mock_agent_file), "--dataset", str(mock_dataset_file)],
        )
        assert result.exit_code != 0
        assert "Unknown metric" in result.output


def test_tune_compile_fail(runner: CliRunner, mock_agent_file: Path, mock_dataset_file: Path) -> None:
    with patch("coreason_optimizer.main.OpenAIClient"), patch("coreason_optimizer.main.MiproOptimizer") as MockOpt:
        MockOpt.return_value.compile.side_effect = Exception("Compile error")

        result = runner.invoke(
            cli,
            ["tune", "--agent", str(mock_agent_file), "--dataset", str(mock_dataset_file)],
        )
        assert result.exit_code != 0
        assert "Optimization failed: Compile error" in result.output


def test_tune_save_fail(runner: CliRunner, mock_agent_file: Path, mock_dataset_file: Path, tmp_path: Path) -> None:
    # Use a directory as output file to trigger error
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    with patch("coreason_optimizer.main.OpenAIClient"), patch("coreason_optimizer.main.MiproOptimizer") as MockOpt:
        MockOpt.return_value.compile.return_value = OptimizedManifest(
            agent_id="t",
            base_model="m",
            optimized_instruction="i",
            few_shot_examples=[],
            performance_metric=0,
            optimization_run_id="id",
        )

        result = runner.invoke(
            cli,
            ["tune", "--agent", str(mock_agent_file), "--dataset", str(mock_dataset_file), "--output", str(out_dir)],
        )
        assert result.exit_code != 0
        # Windows raises Permission denied, Linux raises Is a directory
        assert "Is a directory" in result.output or "Permission denied" in result.output


def test_evaluate_client_fail(runner: CliRunner, mock_manifest_file: Path, mock_dataset_file: Path) -> None:
    with patch("coreason_optimizer.main.OpenAIClient", side_effect=Exception("No API")):
        result = runner.invoke(
            cli,
            ["evaluate", "--manifest", str(mock_manifest_file), "--dataset", str(mock_dataset_file)],
        )
        assert result.exit_code != 0
        assert "Failed to initialize OpenAI Client" in result.output


def test_evaluate_metric_fail(runner: CliRunner, mock_manifest_file: Path, mock_dataset_file: Path) -> None:
    with (
        patch("coreason_optimizer.main.OpenAIClient"),
        patch("coreason_optimizer.main.MetricFactory.get", side_effect=ValueError("Bad metric")),
    ):
        result = runner.invoke(
            cli,
            ["evaluate", "--manifest", str(mock_manifest_file), "--dataset", str(mock_dataset_file)],
        )
        assert result.exit_code != 0
        assert "Bad metric" in result.output


def test_tune_invalid_format(runner: CliRunner, mock_agent_file: Path, tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.touch()
    result = runner.invoke(cli, ["tune", "--agent", str(mock_agent_file), "--dataset", str(p)])
    assert result.exit_code != 0
    assert "Unsupported file format" in result.output


def test_evaluate_success(runner: CliRunner, mock_manifest_file: Path, mock_dataset_file: Path) -> None:
    with patch("coreason_optimizer.main.OpenAIClient") as MockClient:
        mock_client_instance = MockClient.return_value
        # Mock generate response
        mock_response = MagicMock()
        mock_response.content = "hello"  # Matches dataset reference
        mock_client_instance.generate.return_value = mock_response

        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--manifest",
                str(mock_manifest_file),
                "--dataset",
                str(mock_dataset_file),
            ],
        )

        assert result.exit_code == 0
        assert "Evaluation Complete" in result.output
        # Assuming exact_match and "hello"=="hello", score should be 1.0 (average of 1.0 and 0.0 depending on mocks)
        # Dataset has "hello" and "goodbye". Mock returns "hello" always.
        # 1st: hello == hello -> 1.0
        # 2nd: hello != goodbye -> 0.0
        # Avg: 0.5
        assert "0.5000" in result.output


def test_evaluate_manifest_missing(runner: CliRunner, mock_dataset_file: Path) -> None:
    result = runner.invoke(cli, ["evaluate", "--manifest", "missing.json", "--dataset", str(mock_dataset_file)])
    assert result.exit_code != 0
    assert "Failed to load manifest" in result.output


def test_evaluate_dataset_missing(runner: CliRunner, mock_manifest_file: Path) -> None:
    result = runner.invoke(cli, ["evaluate", "--manifest", str(mock_manifest_file), "--dataset", "missing.jsonl"])
    assert result.exit_code != 0
    assert "Failed to load dataset" in result.output


def test_evaluate_csv_fallback(runner: CliRunner, mock_manifest_file: Path, tmp_path: Path) -> None:
    # Create valid CSV
    p = tmp_path / "data.csv"
    p.write_text("q,reference\nval1,ref1", encoding="utf-8")

    with patch("coreason_optimizer.main.OpenAIClient") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.content = "ref1"
        mock_client_instance.generate.return_value = mock_response

        result = runner.invoke(
            cli,
            ["evaluate", "--manifest", str(mock_manifest_file), "--dataset", str(p)],
        )
        assert result.exit_code == 0
        assert "Evaluation Complete" in result.output


def test_evaluate_loop_exception(runner: CliRunner, mock_manifest_file: Path, mock_dataset_file: Path) -> None:
    with patch("coreason_optimizer.main.OpenAIClient") as MockClient:
        # Mock generate to raise exception
        MockClient.return_value.generate.side_effect = Exception("Generate error")

        result = runner.invoke(
            cli,
            ["evaluate", "--manifest", str(mock_manifest_file), "--dataset", str(mock_dataset_file)],
        )
        assert result.exit_code == 0
        assert "Evaluation Complete" in result.output
        # Score should be 0.0 because all failed
        assert "0.0000" in result.output


def test_evaluate_csv_fail_no_demos(runner: CliRunner, tmp_path: Path) -> None:
    # Manifest without examples
    m_path = tmp_path / "no_demos.json"
    manifest = OptimizedManifest(
        agent_id="test",
        base_model="gpt-4o",
        optimized_instruction="sys",
        few_shot_examples=[],  # Empty
        performance_metric=1.0,
        optimization_run_id="123",
    )
    m_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    d_path = tmp_path / "data.csv"
    d_path.write_text("q,reference\nval1,ref1", encoding="utf-8")

    result = runner.invoke(
        cli,
        ["evaluate", "--manifest", str(m_path), "--dataset", str(d_path)],
    )
    assert result.exit_code != 0
    assert "Cannot infer CSV schema" in result.output
