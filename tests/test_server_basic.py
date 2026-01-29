import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock

# Set dummy API key for tests
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"

from coreason_optimizer.server import app
from coreason_optimizer.core.models import TrainingExample, OptimizedManifest

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

def test_optimize_schema_validation():
    with TestClient(app) as client:
        # Invalid request (missing fields)
        response = client.post("/optimize", json={})
        assert response.status_code == 422

@patch("coreason_optimizer.server.MiproOptimizer")
@patch("coreason_optimizer.server.BootstrapFewShot")
@patch("coreason_optimizer.server.MetricFactory")
def test_optimize_endpoint(mock_metric, mock_bootstrap, mock_mipro):
    # Setup mocks
    mock_optimizer_instance = MagicMock()
    mock_mipro.return_value = mock_optimizer_instance

    mock_manifest = OptimizedManifest(
        agent_id="test_agent",
        base_model="gpt-4o",
        optimized_instruction="Optimized system prompt",
        few_shot_examples=[],
        performance_metric=1.0,
        optimization_run_id="test_run"
    )
    mock_optimizer_instance.compile.return_value = mock_manifest

    payload = {
        "agent": {
            "system_prompt": "Original prompt",
            "inputs": ["input1"],
            "outputs": ["output1"]
        },
        "dataset": [
            {
                "inputs": {"input1": "val1"},
                "reference": "ref1",
                "metadata": {}
            },
            {
                "inputs": {"input1": "val2"},
                "reference": "ref2",
                "metadata": {}
            }
        ],
        "config": {
            "metric": "exact_match",
            "target_model": "gpt-4o"
        },
        "strategy": "mipro"
    }

    with TestClient(app) as client:
        response = client.post("/optimize", json=payload)

        # If the response is not 200, print details
        if response.status_code != 200:
            print(response.json())

        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test_agent"
        assert data["optimized_instruction"] == "Optimized system prompt"

        # Verify Mipro was called
        mock_mipro.assert_called_once()
        mock_optimizer_instance.compile.assert_called_once()

def test_dynamic_construct():
    from coreason_optimizer.server import DynamicConstruct
    from coreason_optimizer.server_schemas import AgentDefinition

    ad = AgentDefinition(
        system_prompt="sys",
        inputs=["i"],
        outputs=["o"]
    )
    dc = DynamicConstruct(ad)
    assert dc.system_prompt == "sys"
    assert dc.inputs == ["i"]
    assert dc.outputs == ["o"]

@pytest.mark.asyncio
async def test_bridged_client():
    from coreason_optimizer.server import BridgedLLMClient
    import anyio

    mock_async = AsyncMock()
    mock_async.generate.return_value = "response"

    bridge = BridgedLLMClient(mock_async)

    def worker():
        return bridge.generate(messages=[])

    result = await anyio.to_thread.run_sync(worker)
    assert result == "response"
    mock_async.generate.assert_called_once()

@pytest.mark.asyncio
async def test_bridged_embedding_provider():
    from coreason_optimizer.server import BridgedEmbeddingProvider
    import anyio

    mock_async = AsyncMock()
    mock_async.embed.return_value = "embeddings"

    bridge = BridgedEmbeddingProvider(mock_async)

    def worker():
        return bridge.embed(texts=[])

    result = await anyio.to_thread.run_sync(worker)
    assert result == "embeddings"
    mock_async.embed.assert_called_once()

def test_optimize_errors_and_bootstrap():
    payload = {
        "agent": {
            "system_prompt": "Original prompt",
            "inputs": ["input1"],
            "outputs": ["output1"]
        },
        "dataset": [
            {
                "inputs": {"input1": "val1"},
                "reference": "ref1",
                "metadata": {}
            }
        ],
        "config": {
            "metric": "unknown_metric",
            "target_model": "gpt-4o"
        },
        "strategy": "mipro"
    }

    # 1. Unknown metric
    with TestClient(app) as client:
        response = client.post("/optimize", json=payload)
        assert response.status_code == 400
        assert "Unknown metric" in response.text

    # 2. Bootstrap strategy
    payload["config"]["metric"] = "exact_match"
    payload["strategy"] = "bootstrap"

    with patch("coreason_optimizer.server.BootstrapFewShot") as mock_boot:
        mock_instance = MagicMock()
        mock_boot.return_value = mock_instance
        # Mock compile return
        mock_instance.compile.return_value = OptimizedManifest(
             agent_id="test", base_model="gpt", optimized_instruction="sys",
             performance_metric=1.0, optimization_run_id="id"
        )

        with TestClient(app) as client:
            response = client.post("/optimize", json=payload)
            assert response.status_code == 200
            mock_boot.assert_called_once()

    # 3. Exception handling
    payload["strategy"] = "mipro"
    with patch("coreason_optimizer.server.MiproOptimizer") as mock_mipro:
        mock_mipro.return_value.compile.side_effect = Exception("Boom")
        with TestClient(app) as client:
            response = client.post("/optimize", json=payload)
            assert response.status_code == 500
            assert "Boom" in response.text

def test_missing_state():
    payload = {
        "agent": {
            "system_prompt": "Original prompt",
            "inputs": ["input1"],
            "outputs": ["output1"]
        },
        "dataset": [
            {
                "inputs": {"input1": "val1"},
                "reference": "ref1",
                "metadata": {}
            }
        ],
        "config": {"metric": "exact_match", "selector_type": "random"},
        "strategy": "mipro"
    }

    with TestClient(app) as client:
        # 1. Missing LLM Client
        # We need to temporarily remove the attr from app.state
        # app.state is available via client.app.state
        llm = client.app.state.llm_client_async
        del client.app.state.llm_client_async

        response = client.post("/optimize", json=payload)
        assert response.status_code == 500
        assert "LLM Client not initialized" in response.text

        # Restore
        client.app.state.llm_client_async = llm

        # 2. Missing Embedding Client (when selector is semantic)
        payload["config"]["selector_type"] = "semantic"
        embed = client.app.state.embedding_client_async
        del client.app.state.embedding_client_async

        response = client.post("/optimize", json=payload)
        assert response.status_code == 500
        assert "Embedding Client not initialized" in response.text

        # Restore
        client.app.state.embedding_client_async = embed
