from unittest.mock import MagicMock

from coreason_optimizer.core.interfaces import EmbeddingResponse, UsageStats
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.selector import SemanticSelector


def test_selector_sync_direct_coverage() -> None:
    # Dedicated test for SemanticSelector Sync lines 152, 167-173
    mock_provider = MagicMock()

    # 1. Short circuit
    selector = SemanticSelector(mock_provider, seed=42)
    ds = Dataset([TrainingExample(inputs={"q": str(i)}, reference="A") for i in range(1)])

    res = selector.select(ds, k=5)
    assert len(res) == 1
    mock_provider.embed.assert_not_called()

    # 2. Clustering
    # Need distinct points to avoid convergence warning issues potentially affecting flow?
    # Ensure k < len
    ds_large = Dataset([TrainingExample(inputs={"q": str(i)}, reference="A") for i in range(5)])

    # Mock embedding response with distinct vectors
    # 5 vectors of dim 2
    embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5]]
    mock_provider.embed.return_value = EmbeddingResponse(embeddings=embeddings, usage=UsageStats())

    res = selector.select(ds_large, k=2)
    assert len(res) == 2
    mock_provider.embed.assert_called_once()
