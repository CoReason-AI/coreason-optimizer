from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_optimizer.core.async_client import OpenAIClientAsync
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import EmbeddingResponse, LLMResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.bootstrap import BootstrapFewShot, BootstrapFewShotAsync
from coreason_optimizer.strategies.mutator import LLMInstructionMutator, LLMInstructionMutatorAsync
from coreason_optimizer.strategies.selector import SemanticSelector


# Async Client coverage
@pytest.mark.asyncio
async def test_async_client_lifecycle_explicit() -> None:
    # Force cover lines 63-64
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_instance = MockOpenAI.return_value
        mock_instance.close = AsyncMock()

        client = OpenAIClientAsync(api_key="test")
        await client.__aenter__()
        # Internal client is True
        await client.__aexit__(None, None, None)

        mock_instance.close.assert_awaited()


@pytest.mark.asyncio
async def test_async_client_aexit_exception() -> None:
    # Test aexit when exception occurred
    with patch("coreason_optimizer.core.async_client.AsyncOpenAI") as MockOpenAI:
        mock_instance = MockOpenAI.return_value
        mock_instance.close = AsyncMock()

        try:
            async with OpenAIClientAsync(api_key="test"):
                raise ValueError("Boom")
        except ValueError:
            pass

        mock_instance.close.assert_awaited()


# Bootstrap Exception coverage
@pytest.mark.asyncio
async def test_bootstrap_exception_swallow_explicit() -> None:
    # Force cover line 135
    mock_client = AsyncMock()
    metric = MagicMock()
    config = OptimizerConfig()
    bs = BootstrapFewShotAsync(mock_client, metric, config)

    agent = MagicMock()
    agent.system_prompt = "sys"
    train = [TrainingExample(inputs={"q": "1"}, reference="A")]
    val = [TrainingExample(inputs={"q": "2"}, reference="B")]

    # First call success (mining), second call raise Exception (validation)
    mock_client.generate.side_effect = [MagicMock(content="pred", usage=UsageStats()), Exception("Generic Error")]

    metric.return_value = 1.0

    manifest = await bs.compile(agent, train, val)
    assert manifest.performance_metric == 0.0


def test_bootstrap_sync_compile_exception() -> None:
    # Cover Sync Facade exception handling (lines 192+)
    mock_client = MagicMock()
    # Mock adapter's compile to raise Exception
    # BootstrapFewShot uses Adapter if not async

    bs = BootstrapFewShot(mock_client, MagicMock(), OptimizerConfig())
    # We need to mock the async implementation on the instance
    # But bs._async is created in init.

    # We can mock anyio.run to raise exception?
    with patch("coreason_optimizer.strategies.bootstrap.anyio.run", side_effect=Exception("Run Error")):
        with pytest.raises(Exception, match="Run Error"):
            bs.compile(MagicMock(), [], [])


# Sync Mutator coverage
def test_mutator_sync_full_coverage() -> None:
    client = MagicMock()
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutator(client, config)

    # 1. Exception handling (lines 150-155)
    client.generate.side_effect = Exception("error")
    res = mutator.mutate("instr", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert res == "instr"

    # 2. Empty response (lines 158-159)
    client.generate.side_effect = None
    client.generate.return_value = LLMResponse(content="", usage=UsageStats())
    res = mutator.mutate("instr", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert res == "instr"

    # 3. Markdown cleanup (lines 164-166)
    client.generate.return_value = LLMResponse(content="```\nnew\n```", usage=UsageStats())
    res = mutator.mutate("instr", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert res == "new"


# Async Mutator coverage
@pytest.mark.asyncio
async def test_mutator_async_full_coverage() -> None:
    client = AsyncMock()
    config = OptimizerConfig(meta_model="meta")
    mutator = LLMInstructionMutatorAsync(client, config)

    # 1. Exception handling
    client.generate.side_effect = Exception("error")
    res = await mutator.mutate("instr", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert res == "instr"

    # 2. Empty response
    client.generate.side_effect = None
    client.generate.return_value = LLMResponse(content="", usage=UsageStats())
    res = await mutator.mutate("instr", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert res == "instr"

    # 3. Markdown cleanup
    client.generate.return_value = LLMResponse(content="```\nnew\n```", usage=UsageStats())
    res = await mutator.mutate("instr", [TrainingExample(inputs={"q": "1"}, reference="A")])
    assert res == "new"


# Selector coverage
def test_selector_sync_full_coverage() -> None:
    provider = MagicMock()
    selector = SemanticSelector(provider, seed=42)
    ds = Dataset([TrainingExample(inputs={"q": str(i)}, reference="A") for i in range(5)])

    # 1. Short circuit (line 152)
    res = selector.select(ds, k=10)
    assert len(res) == 5
    provider.embed.assert_not_called()

    # 2. Clustering (lines 167-173)
    provider.embed.return_value = EmbeddingResponse(embeddings=[[0.1] * 1536] * 5, usage=UsageStats())
    res = selector.select(ds, k=2)
    assert len(res) == 2
    provider.embed.assert_called()
