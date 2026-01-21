# coreason-optimizer

**Automated Prompt Engineering / LLM Compilation / DSPy Integration for CoReason-AI**

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI Status](https://github.com/CoReason-AI/coreason-optimizer/actions/workflows/main.yml/badge.svg)](https://github.com/CoReason-AI/coreason-optimizer/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-product_requirements-blue)](docs/product_requirements.md)

**coreason-optimizer** is the "Compiler" for the CoReason Agentic Platform. It automates prompt engineering by treating prompts as trainable weights, optimizing them against ground-truth datasets to maximize performance metrics.

---

## Installation

```bash
pip install coreason-optimizer
```

## Features

-   **Automated Optimization:** Rewrites instructions and selects examples to maximize a score, not human intuition.
-   **Model-Specific Compilation:** Generates optimized prompts specifically tuned for target models (e.g., GPT-4, Claude 3.5).
-   **Continuous Learning:** Re-runs optimization on recent logs to patch prompts against data drift.
-   **Mutate-Evaluate Loop:** Systematic cycle of drafting, evaluating, diagnosing, mutating, and selecting prompts.
-   **Strategies:** Includes BootstrapFewShot (mining successful traces) and MIPRO (Multi-prompt Instruction PRoposal Optimizer).
-   **Integration:** Works seamlessly with `coreason-construct`, `coreason-archive`, and `coreason-assay`.

For full product requirements, see [docs/product_requirements.md](docs/product_requirements.md).

## Usage

Here is how to initialize and use the library to compile an agent:

```python
from coreason_optimizer import OptimizerConfig, PromptOptimizer
from coreason_optimizer.core.interfaces import Construct
from coreason_optimizer.data import Dataset

# 1. Configuration
config = OptimizerConfig(
    target_model="gpt-4o",
    metric="exact_match",
    max_rounds=10
)

# 2. Load Data
dataset = Dataset.from_csv("data/gold_set.csv")
train_set, val_set = dataset.split(test_size=0.2)

# 3. Load Agent (Construct)
# In a real scenario, this would be imported from your agent code
# from src.agents.analyst import analyst_agent
class MockAgent(Construct):
    inputs = ["question"]
    outputs = ["answer"]
    system_prompt = "You are a helpful assistant."
agent = MockAgent()

# 4. Compile
optimizer = PromptOptimizer(config=config)
optimized_manifest = optimizer.compile(
    agent=agent,
    trainset=train_set,
    valset=val_set
)

print(f"Optimization complete. New Score: {optimized_manifest.performance_metric}")
print(f"Optimized Instruction: {optimized_manifest.optimized_instruction}")
```