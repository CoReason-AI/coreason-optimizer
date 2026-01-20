# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import sys
from unittest.mock import MagicMock, patch


def test_logger_creates_directory_if_missing():
    # Remove logger from sys.modules to force re-import
    if "coreason_optimizer.utils.logger" in sys.modules:
        del sys.modules["coreason_optimizer.utils.logger"]

    # We need to patch pathlib.Path because the module uses `from pathlib import Path`
    # and executes code at module level.
    with patch("pathlib.Path") as MockPath:
        # Mock instance
        mock_path_instance = MagicMock()
        MockPath.return_value = mock_path_instance

        # Setup: Path("logs") -> mock_path_instance
        # We need to make sure subsequent calls (if any) behave sanely,
        # but the module only calls Path("logs") and "logs/app.log" (in logger.add)

        # Simulate directory does NOT exist
        mock_path_instance.exists.return_value = False

        # Import the module

        # Verify mkdir was called
        # The module calls Path("logs")
        # Then calls exists() on it
        # Then calls mkdir()
        mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Cleanup: remove from sys.modules again to prevent using the mocked version later
    if "coreason_optimizer.utils.logger" in sys.modules:
        del sys.modules["coreason_optimizer.utils.logger"]
