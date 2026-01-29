# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

"""
coreason-optimizer package initialization.

This package provides the "Compiler" for the CoReason Agentic Platform,
treating prompts as trainable weights and optimizing them against ground-truth datasets.
"""

from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.interfaces import PromptOptimizer

__version__ = "0.2.0"
__author__ = "Gowtham A Rao"
__email__ = "gowtham.rao@coreason.ai"

__all__ = ["OptimizerConfig", "PromptOptimizer"]
