# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from coreason_optimizer.core.models import OptimizedManifest, TrainingExample


def test_training_example_creation() -> None:
    """Test creating a TrainingExample."""
    example = TrainingExample(
        inputs={"question": "What is 2+2?"},
        reference="4",
        metadata={"source": "math_dataset"},
    )
    assert example.inputs["question"] == "What is 2+2?"
    assert example.reference == "4"
    assert example.metadata["source"] == "math_dataset"


def test_training_example_defaults() -> None:
    """Test default values for TrainingExample."""
    example = TrainingExample(inputs={"q": "a"}, reference="b")
    assert example.metadata == {}


def test_optimized_manifest_creation() -> None:
    """Test creating an OptimizedManifest."""
    example = TrainingExample(inputs={"q": "foo"}, reference="bar")
    manifest = OptimizedManifest(
        agent_id="test_agent",
        base_model="gpt-4o",
        optimized_instruction="You are a helpful assistant.",
        few_shot_examples=[example],
        performance_metric=0.95,
        optimization_run_id="run_123",
    )

    assert manifest.agent_id == "test_agent"
    assert manifest.performance_metric == 0.95
    assert len(manifest.few_shot_examples) == 1
    assert manifest.few_shot_examples[0].inputs["q"] == "foo"


def test_manifest_serialization() -> None:
    """Test JSON serialization of the manifest."""
    example = TrainingExample(inputs={"q": "foo"}, reference="bar")
    manifest = OptimizedManifest(
        agent_id="test_agent",
        base_model="gpt-4o",
        optimized_instruction="prompt",
        few_shot_examples=[example],
        performance_metric=1.0,
        optimization_run_id="run_1",
    )

    json_str = manifest.model_dump_json()
    assert "test_agent" in json_str
    assert "foo" in json_str
    assert "bar" in json_str
