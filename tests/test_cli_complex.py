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
    p = d / "agent.py"
    p.write_text(
        "class A:\n  system_prompt='s'\n  inputs=['i']\n  outputs=['o']\nagent=A()",
        encoding="utf-8",
    )
    return p


def test_tune_empty_dataset(
    runner: CliRunner, mock_agent_file: Path, tmp_path: Path
) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")

    with patch("coreason_optimizer.main.OpenAIClient"), patch(
        "coreason_optimizer.main.MiproOptimizer"
    ) as MockOpt:

        MockOpt.return_value.compile.return_value = OptimizedManifest(
            agent_id="test",
            base_model="gpt-4o",
            optimized_instruction="sys",
            few_shot_examples=[],
            performance_metric=0.0,
            optimization_run_id="run",
        )

        result = runner.invoke(
            cli, ["tune", "--agent", str(mock_agent_file), "--dataset", str(p)]
        )
        assert result.exit_code == 0
        assert "Optimization complete" in result.output


def test_tune_output_parent_missing(
    runner: CliRunner, mock_agent_file: Path, tmp_path: Path
) -> None:
    p = tmp_path / "data.jsonl"
    p.write_text('{"input":{"i":"1"},"output":"2"}\n', encoding="utf-8")

    output = tmp_path / "missing_dir" / "out.json"
    # missing_dir does not exist

    with patch("coreason_optimizer.main.OpenAIClient"), patch(
        "coreason_optimizer.main.MiproOptimizer"
    ) as MockOpt:

        MockOpt.return_value.compile.return_value = OptimizedManifest(
            agent_id="test",
            base_model="gpt-4o",
            optimized_instruction="sys",
            few_shot_examples=[],
            performance_metric=0.0,
            optimization_run_id="run",
        )

        result = runner.invoke(
            cli,
            [
                "tune",
                "--agent",
                str(mock_agent_file),
                "--dataset",
                str(p),
                "--output",
                str(output),
            ],
        )
        assert result.exit_code != 0
        # Error message usually contains "No such file or directory" or "FileNotFoundError"
        # On Windows/Linux it might vary slightly but "No such file or directory" is common in str(e)
        assert (
            "No such file or directory" in result.output
            or "FileNotFoundError" in result.output
        )


def test_integration_tune_evaluate(
    runner: CliRunner, mock_agent_file: Path, tmp_path: Path
) -> None:
    # Data
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"input":{"i":"q"},"reference":"a"}\n', encoding="utf-8")

    output_manifest = tmp_path / "manifest.json"

    # Mocks
    with patch("coreason_optimizer.main.OpenAIClient") as MockClient, patch(
        "coreason_optimizer.main.MiproOptimizer"
    ) as MockOpt:

        # Tune Step
        manifest = OptimizedManifest(
            agent_id="test",
            base_model="gpt-4o",
            optimized_instruction="optimized_sys",
            few_shot_examples=[
                TrainingExample(inputs={"i": "ex_q"}, reference="ex_a")
            ],
            performance_metric=1.0,
            optimization_run_id="run_1",
        )
        MockOpt.return_value.compile.return_value = manifest

        result_tune = runner.invoke(
            cli,
            [
                "tune",
                "--agent",
                str(mock_agent_file),
                "--dataset",
                str(data_file),
                "--output",
                str(output_manifest),
            ],
        )
        assert result_tune.exit_code == 0
        assert output_manifest.exists()

        # Evaluate Step
        # Mock generate for evaluation
        mock_resp = MagicMock()
        mock_resp.content = "a"  # Match reference
        MockClient.return_value.generate.return_value = mock_resp

        result_eval = runner.invoke(
            cli,
            [
                "evaluate",
                "--manifest",
                str(output_manifest),
                "--dataset",
                str(data_file),
            ],
        )
        assert result_eval.exit_code == 0
        assert "Evaluation Complete" in result_eval.output
        assert "1.0000" in result_eval.output


def test_evaluate_f1_score(runner: CliRunner, tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest = OptimizedManifest(
        agent_id="test",
        base_model="gpt-4o",
        optimized_instruction="optimized_sys",
        few_shot_examples=[],
        performance_metric=1.0,
        optimization_run_id="run_1",
    )
    manifest_file.write_text(manifest.model_dump_json(), encoding="utf-8")

    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        '{"input":{"i":"q"},"reference":"token match"}\n', encoding="utf-8"
    )

    with patch("coreason_optimizer.main.OpenAIClient") as MockClient:
        # Prediction has partial overlap "token"
        mock_resp = MagicMock()
        mock_resp.content = "token mismatch"
        MockClient.return_value.generate.return_value = mock_resp

        # F1 between "token match" and "token mismatch":
        # Tokens: {token, match} vs {token, mismatch}. Common: {token}.
        # Prec: 1/2. Rec: 1/2. F1: 0.5.

        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--manifest",
                str(manifest_file),
                "--dataset",
                str(data_file),
                "--metric",
                "f1_score",
            ],
        )
        assert result.exit_code == 0
        assert "f1_score" in result.output
        assert "0.5000" in result.output
