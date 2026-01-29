from pathlib import Path

import pytest
from coreason_identity.models import UserContext

from coreason_optimizer.core.client import OptimizationClient
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import StrategySelector


def test_optimization_client_enforcement() -> None:
    client = OptimizationClient()

    # Should fail without context
    with pytest.raises(ValueError, match="UserContext is required"):
        client.register_study("test-study", context=None)

    # Should pass with context
    context = UserContext(user_id="test-user", email="test@example.com")

    study_id = client.register_study("test-study", context=context)
    assert study_id.startswith("study_")

    # Test get_suggestion
    with pytest.raises(ValueError, match="UserContext is required"):
        client.get_suggestion("study_1", context=None)

    sugg = client.get_suggestion("study_1", context=context)
    assert isinstance(sugg, dict)
    assert "param_a" in sugg

    # Test report_metric
    with pytest.raises(ValueError, match="UserContext is required"):
        client.report_metric("study_1", 1.0, context=None)

    client.report_metric("study_1", 1.0, context=context)
    # Also test reporting to an existing study to cover that branch
    client.report_metric(study_id, 0.95, context=context)


def test_strategy_selector_enforcement() -> None:
    selector = StrategySelector()

    # Should fail without context
    with pytest.raises(ValueError, match="UserContext is required"):
        selector.select_strategy("mipro", context=None)

    # Should pass with context
    context = UserContext(user_id="test-user", email="test@example.com")
    strategy = selector.select_strategy("mipro", context=context)
    assert strategy == "mipro"


def test_dataset_loader_enforcement(tmp_path: Path) -> None:
    # Create dummy csv
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("input,reference\na,b\n", encoding="utf-8")

    # Should fail without context
    with pytest.raises(ValueError, match="UserContext is required"):
        Dataset.from_csv(csv_file, input_cols=["input"], reference_col="reference", context=None)

    # Should pass with context
    context = UserContext(user_id="test-user", email="test@example.com")
    ds = Dataset.from_csv(csv_file, input_cols=["input"], reference_col="reference", context=context)
    assert len(ds) == 1


def test_dataset_loader_jsonl_enforcement(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"input": "a", "output": "b"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="UserContext is required"):
        Dataset.from_jsonl(jsonl_file, context=None)

    context = UserContext(user_id="test-user", email="test@example.com")
    ds = Dataset.from_jsonl(jsonl_file, context=context)
    assert len(ds) == 1
