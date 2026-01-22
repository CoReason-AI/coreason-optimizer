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

from coreason_optimizer.core.interfaces import LLMResponse
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
        f.write(json.dumps({"input": "q2", "reference": "a2"}) + "\n")
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


def test_cli_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "coreason-opt" in result.output


def test_tune_bootstrap_success(
    runner: CliRunner, mock_agent_file: str, mock_dataset_file: str, tmp_path: MagicMock
) -> None:
    output_file = str(tmp_path / "out.json")
    with patch("coreason_optimizer.main.load_agent_from_path") as mock_load:
        mock_construct = MagicMock()
        mock_construct.system_prompt = "Sys"
        mock_construct.inputs = ["input"]
        mock_load.return_value = mock_construct

        with patch("coreason_optimizer.main.OpenAIClient") as MockClient:
            mock_client_inst = MagicMock()
            mock_client_inst.generate.return_value = LLMResponse(content="a1", usage={}, cost_usd=0.0)
            MockClient.return_value = mock_client_inst

            with patch("coreason_optimizer.strategies.bootstrap.BootstrapFewShot.compile") as mock_compile:
                mock_compile.return_value = OptimizedManifest(
                    agent_id="test",
                    base_model="gpt-4",
                    optimized_instruction="Opt",
                    few_shot_examples=[],
                    performance_metric=1.0,
                    optimization_run_id="1",
                )

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
                        output_file,
                    ],
                )

                assert result.exit_code == 0
                assert "Optimization complete" in result.output


def test_tune_mipro_success(
    runner: CliRunner, mock_agent_file: str, mock_dataset_file: str, tmp_path: MagicMock
) -> None:
    output_file = str(tmp_path / "out_mipro.json")
    with patch("coreason_optimizer.main.load_agent_from_path") as mock_load:
        mock_construct = MagicMock()
        mock_load.return_value = mock_construct

        with patch("coreason_optimizer.main.OpenAIClient"):
            # If selector is semantic, we need embedding client
            with patch("coreason_optimizer.main.OpenAIEmbeddingClient"):
                pass

            with patch("coreason_optimizer.strategies.mipro.MiproOptimizer.compile") as mock_compile:
                mock_compile.return_value = OptimizedManifest(
                    agent_id="test",
                    base_model="gpt-4",
                    optimized_instruction="Opt",
                    few_shot_examples=[],
                    performance_metric=1.0,
                    optimization_run_id="1",
                )

                result = runner.invoke(
                    cli,
                    [
                        "tune",
                        "--agent",
                        mock_agent_file,
                        "--dataset",
                        mock_dataset_file,
                        "--strategy",
                        "mipro",
                        "--output",
                        output_file,
                    ],
                )

                assert result.exit_code == 0


def test_tune_semantic_selector(
    runner: CliRunner, mock_agent_file: str, mock_dataset_file: str, tmp_path: MagicMock
) -> None:
    output_file = str(tmp_path / "out_sem.json")
    with patch("coreason_optimizer.main.load_agent_from_path") as mock_load:
        mock_construct = MagicMock()
        mock_load.return_value = mock_construct

        with patch("coreason_optimizer.main.OpenAIClient"):
            with patch("coreason_optimizer.main.OpenAIEmbeddingClient") as MockEmbed:
                with patch("coreason_optimizer.strategies.mipro.MiproOptimizer.compile") as mock_compile:
                    mock_compile.return_value = OptimizedManifest(
                        agent_id="test",
                        base_model="gpt-4",
                        optimized_instruction="Opt",
                        few_shot_examples=[],
                        performance_metric=1.0,
                        optimization_run_id="1",
                    )

                    result = runner.invoke(
                        cli,
                        [
                            "tune",
                            "--agent",
                            mock_agent_file,
                            "--dataset",
                            mock_dataset_file,
                            "--strategy",
                            "mipro",
                            "--selector",
                            "semantic",
                            "--output",
                            output_file,
                        ],
                    )
                    assert result.exit_code == 0
                    MockEmbed.assert_called_once()


def test_tune_semantic_selector_fail_init(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    with patch("coreason_optimizer.main.load_agent_from_path") as mock_load:
        mock_construct = MagicMock()
        mock_load.return_value = mock_construct

        with patch("coreason_optimizer.main.OpenAIClient"):
            with patch("coreason_optimizer.main.OpenAIEmbeddingClient", side_effect=Exception("Embed Error")):
                result = runner.invoke(
                    cli,
                    [
                        "tune",
                        "--agent",
                        mock_agent_file,
                        "--dataset",
                        mock_dataset_file,
                        "--strategy",
                        "mipro",
                        "--selector",
                        "semantic",
                        "--output",
                        "out_sem.json",
                    ],
                )
                assert result.exit_code != 0
                assert "Failed to initialize OpenAI Embedding Client" in result.output


def test_tune_fail_load_agent(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    with patch("coreason_optimizer.main.load_agent_from_path", side_effect=Exception("Load Error")):
        result = runner.invoke(cli, ["tune", "--agent", mock_agent_file, "--dataset", mock_dataset_file])
        assert result.exit_code != 0
        # Check for part of exception message that propagates
        assert "Load Error" in result.output


def test_tune_fail_dataset(runner: CliRunner, mock_agent_file: str) -> None:
    with patch("coreason_optimizer.main.load_agent_from_path"):
        result = runner.invoke(cli, ["tune", "--agent", mock_agent_file, "--dataset", "missing.jsonl"])
        assert result.exit_code != 0
        # Dataset.from_jsonl raises FileNotFoundError
        # ClickException format: Error: ...
        assert "missing.jsonl" in result.output or "File not found" in result.output


def test_tune_fail_dataset_invalid_ext(runner: CliRunner, mock_agent_file: str) -> None:
    with patch("coreason_optimizer.main.load_agent_from_path"):
        result = runner.invoke(cli, ["tune", "--agent", mock_agent_file, "--dataset", "invalid.txt"])
        assert result.exit_code != 0
        assert "Unsupported file format" in result.output


def test_tune_fail_client_init(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    with patch("coreason_optimizer.main.load_agent_from_path"):
        with patch("coreason_optimizer.main.OpenAIClient", side_effect=Exception("Auth Error")):
            result = runner.invoke(cli, ["tune", "--agent", mock_agent_file, "--dataset", mock_dataset_file])
            assert result.exit_code != 0
            assert "Failed to initialize OpenAI Client" in result.output


def test_tune_compile_fail(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    with patch("coreason_optimizer.main.load_agent_from_path"):
        with patch("coreason_optimizer.main.OpenAIClient"):
            with patch(
                "coreason_optimizer.strategies.mipro.MiproOptimizer.compile", side_effect=Exception("Compile Error")
            ):
                result = runner.invoke(cli, ["tune", "--agent", mock_agent_file, "--dataset", mock_dataset_file])
                assert result.exit_code != 0
                assert "Optimization failed" in result.output


def test_evaluate_success(runner: CliRunner, mock_manifest_file: str, mock_dataset_file: str) -> None:
    with patch("coreason_optimizer.main.OpenAIClient") as MockClient:
        mock_client = MagicMock()
        mock_client.generate.return_value = LLMResponse(content="a1", usage={}, cost_usd=0.0)
        MockClient.return_value = mock_client

        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--manifest",
                mock_manifest_file,
                "--dataset",
                mock_dataset_file,
            ],
        )

        assert result.exit_code == 0
        assert "Evaluation Complete" in result.output


def test_evaluate_fail_manifest(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["evaluate", "--manifest", "missing.json", "--dataset", "d.jsonl"])
    assert result.exit_code != 0
    assert "Failed to load manifest" in result.output


def test_evaluate_fail_dataset(runner: CliRunner, mock_manifest_file: str) -> None:
    result = runner.invoke(cli, ["evaluate", "--manifest", mock_manifest_file, "--dataset", "missing.jsonl"])
    assert result.exit_code != 0
    assert "Failed to load dataset" in result.output


def test_evaluate_fail_dataset_csv_inference(runner: CliRunner, mock_manifest_file: str, tmp_path: MagicMock) -> None:
    # Create a CSV without few shot in manifest to infer columns from (if manifest had no examples)
    # But mock_manifest_file HAS examples. So let's create a manifest WITHOUT examples.

    d = tmp_path / "out"
    p = d / "manifest_empty.json"
    manifest = OptimizedManifest(
        agent_id="agent",
        base_model="gpt-4",
        optimized_instruction="Instr",
        few_shot_examples=[],  # EMPTY
        performance_metric=1.0,
        optimization_run_id="run",
    )
    with open(p, "w") as f:
        f.write(manifest.model_dump_json())

    csv_path = tmp_path / "data.csv"
    csv_path.touch()

    result = runner.invoke(cli, ["evaluate", "--manifest", str(p), "--dataset", str(csv_path)])
    assert result.exit_code != 0
    assert "Cannot infer CSV schema" in result.output


def test_evaluate_save_fail(runner: CliRunner, mock_agent_file: str, mock_dataset_file: str) -> None:
    # Testing save failure in tune actually
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
                # Output to invalid path (directory)
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
                # Errno 21 Is a directory
                assert "Is a directory" in result.output or "Failed to save manifest" in result.output


def test_evaluate_metric_error(runner: CliRunner, mock_manifest_file: str, mock_dataset_file: str) -> None:
    with patch("coreason_optimizer.main.OpenAIClient"):
        result = runner.invoke(
            cli, ["evaluate", "--manifest", mock_manifest_file, "--dataset", mock_dataset_file, "--metric", "unknown"]
        )
        assert result.exit_code != 0
        assert "Unknown metric" in result.output


def test_evaluate_client_init_fail(runner: CliRunner, mock_manifest_file: str, mock_dataset_file: str) -> None:
    with patch("coreason_optimizer.main.OpenAIClient", side_effect=Exception("Client Error")):
        result = runner.invoke(cli, ["evaluate", "--manifest", mock_manifest_file, "--dataset", mock_dataset_file])
        assert result.exit_code != 0
        assert "Failed to initialize OpenAI Client" in result.output
