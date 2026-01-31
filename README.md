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
-   **Optimization-as-a-Service:** Run as a microservice API to compile prompts on-demand.
-   **Model-Specific Compilation:** Generates optimized prompts specifically tuned for target models (e.g., GPT-4, Claude 3.5).
-   **Continuous Learning:** Re-runs optimization on recent logs to patch prompts against data drift.
-   **Mutate-Evaluate Loop:** Systematic cycle of drafting, evaluating, diagnosing, mutating, and selecting prompts.
-   **Strategies:** Includes BootstrapFewShot (mining successful traces) and MIPRO (Multi-prompt Instruction PRoposal Optimizer).
-   **Integration:** Works seamlessly with `coreason-construct`, `coreason-archive`, and `coreason-assay`.

For full product requirements, see [docs/product_requirements.md](docs/product_requirements.md).

## Usage

You can use `coreason-optimizer` as a Python library, a CLI tool, or a Microservice.

### 1. Python Library

```python
from coreason_optimizer import OptimizerConfig, PromptOptimizer
from coreason_optimizer.core.interfaces import Construct
from coreason_optimizer.data import Dataset

# Define Agent
class MockAgent(Construct):
    inputs = ["question"]
    outputs = ["answer"]
    system_prompt = "You are a helpful assistant."
agent = MockAgent()

# Compile
dataset = Dataset.from_csv("data/gold_set.csv")
train_set, val_set = dataset.split(train_ratio=0.8)

optimizer = PromptOptimizer(config=OptimizerConfig(target_model="gpt-4o"))
manifest = optimizer.compile(agent, train_set, val_set)

print(f"Optimized Score: {manifest.performance_metric}")
```

### 2. Server Mode (Microservice)

Run the optimizer as a standalone service using Docker:

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY coreason-optimizer:latest
```

Then call the API:

```bash
curl -X POST http://localhost:8000/optimize -d @request.json
```

For detailed instructions, see **[docs/usage.md](docs/usage.md)**.
