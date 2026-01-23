# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import RandomSelector


def test_random_selector_selection() -> None:
    examples = [TrainingExample(inputs={"q": str(i)}, reference=str(i)) for i in range(10)]
    ds = Dataset(examples)

    selector = RandomSelector(seed=42)
    selected = selector.select(ds, k=3)

    assert len(selected) == 3
    # Check if they are actually from the dataset
    for ex in selected:
        assert ex in examples


def test_random_selector_oversized_request() -> None:
    examples = [
        TrainingExample(inputs={"q": "1"}, reference="1"),
        TrainingExample(inputs={"q": "2"}, reference="2"),
    ]
    ds = Dataset(examples)

    selector = RandomSelector()
    selected = selector.select(ds, k=5)

    # Should return all available if k > len(ds)
    assert len(selected) == 2
    assert len(selected) == len(examples)


def test_random_selector_determinism() -> None:
    examples = [TrainingExample(inputs={"q": str(i)}, reference=str(i)) for i in range(20)]
    ds = Dataset(examples)

    s1 = RandomSelector(seed=123)
    s2 = RandomSelector(seed=123)

    sel1 = s1.select(ds, k=5)
    sel2 = s2.select(ds, k=5)

    # Inputs should be identical
    assert [e.inputs for e in sel1] == [e.inputs for e in sel2]
