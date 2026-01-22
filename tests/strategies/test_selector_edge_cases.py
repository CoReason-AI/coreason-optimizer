# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import pytest

from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import RandomSelector


@pytest.mark.asyncio
async def test_select_k_zero() -> None:
    ds = Dataset([TrainingExample(inputs={"a": 1}, reference=1)])
    sel = RandomSelector()
    selected = await sel.select(ds, k=0)
    assert selected == []


@pytest.mark.asyncio
async def test_select_k_negative() -> None:
    ds = Dataset([TrainingExample(inputs={"a": 1}, reference=1)])
    sel = RandomSelector()
    # random.sample raises ValueError for negative k
    with pytest.raises(ValueError):
        await sel.select(ds, k=-1)
