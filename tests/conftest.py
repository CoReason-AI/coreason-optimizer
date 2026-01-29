
import pytest
from coreason_identity.models import UserContext, SecretStr

@pytest.fixture
def mock_context():
    return UserContext(
        user_id="test-user",
        email="test@example.com",
        groups=["test-group"],
        claims={"source": "test"}
    )
