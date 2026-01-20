# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import importlib
import shutil
from pathlib import Path

import coreason_optimizer.utils.logger


def test_logger_creates_directory_coverage() -> None:
    """
    Test that the logger module creates the 'logs' directory if it doesn't exist.
    This is to ensure 100% coverage of the 'if not log_path.exists():' block.
    """
    log_path = Path("logs")

    # 1. Clean up existing logs directory if possible
    # Note: On some systems, this might fail if a file is locked, but on Linux (CI) it usually works.
    if log_path.exists():
        try:
            shutil.rmtree(log_path)
        except OSError:
            # If we can't delete it (e.g. open file), we might skip this test or try another way.
            # But for coverage we really want to hit that line.
            pass

    # 2. Reload the module. This should re-execute the module-level code.
    importlib.reload(coreason_optimizer.utils.logger)

    # 3. Verify directory was created
    assert log_path.exists()
    assert log_path.is_dir()
