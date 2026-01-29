import shutil
from importlib import reload
from pathlib import Path

from coreason_optimizer.utils import logger


def test_logger_directory_creation() -> None:
    """Test that the logger creates the logs directory if it doesn't exist."""
    # Ensure any previous logger handlers are removed to close file handles
    logger.logger.remove()

    # Ensure logs directory does NOT exist initially
    log_path = Path("logs")
    if log_path.exists():
        shutil.rmtree(log_path)

    try:
        # Reload the module to trigger the top-level code execution
        # We need to reload the submodule directly
        reload(logger)

        # Assert directory was created
        assert log_path.exists()
        assert log_path.is_dir()

    finally:
        # Cleanup (optional, but good practice)
        # Remove the handler again to release the file lock
        logger.logger.remove()

        if log_path.exists():
            shutil.rmtree(log_path)

        # Ensure it exists again for other tests if they rely on it (though they shouldn't rely on global state ideally)
        log_path.mkdir(parents=True, exist_ok=True)
