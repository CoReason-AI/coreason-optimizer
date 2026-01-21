# coreason-optimizer

**coreason-optimizer** is the "Compiler" for the CoReason Agentic Platform. It automates prompt engineering by treating prompts (instructions and few-shot examples) as **trainable weights**.

[![CI/CD](https://github.com/CoReason-AI/coreason_optimizer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason_optimizer/actions/workflows/ci-cd.yml)
[![PyPI](https://img.shields.io/pypi/v/coreason_optimizer.svg)](https://pypi.org/project/coreason_optimizer/)
[![License](https://img.shields.io/github/license/CoReason-AI/coreason_optimizer)](https://github.com/CoReason-AI/coreason_optimizer/blob/main/LICENSE)

## Overview

In the current State-of-the-Art, writing static prompts by hand is technical debt. `coreason-optimizer` solves this by:
*   **Automated Optimization**: Using meta-algorithms to rewrite instructions and select examples.
*   **Model-Specific Compilation**: Tuning prompts specifically for the target model (e.g., GPT-4o, Claude 3.5).
*   **Continuous Learning**: Updating prompts based on new data.

## Features

*   **Strategies**:
    *   **Bootstrap Few-Shot**: Mines successful traces from the training set to create few-shot examples.
    *   **MIPRO (Multi-prompt Instruction PRoposal Optimizer)**: A Bayesian-style optimizer that proposes new instructions and selects the best combination of instruction and few-shot examples.
*   **Cost Awareness**: Built-in `BudgetManager` to halt optimization if token spend exceeds a defined limit.
*   **Data Support**: Loaders for CSV and JSONL datasets with automatic splitting.

## Installation

### Prerequisites

*   Python 3.12+
*   Poetry

### Steps

1.  Clone the repository:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_optimizer.git
    cd coreason_optimizer
    ```
2.  Install dependencies:
    ```sh
    poetry install
    ```

3.  Set up environment variables (required for OpenAI models):
    ```sh
    export OPENAI_API_KEY="sk-..."
    ```

## CLI Usage

The package provides a CLI tool `coreason-opt` for easy integration into CI/CD pipelines.

### Tuning (Optimization)

Optimize an agent's prompt against a dataset.

```sh
poetry run coreason-opt tune \
    --agent src/agents/analyst.py:AnalystAgent \
    --dataset data/gold_set.csv \
    --strategy mipro \
    --output optimized_manifest.json
```

**Options:**
*   `--agent`: Path to the agent file and class/object (e.g., `path/to/file.py:AgentClass`).
*   `--dataset`: Path to the dataset (`.csv` or `.jsonl`).
*   `--strategy`: Optimization strategy: `mipro` (default) or `bootstrap`.
*   --`selector`: Example selection strategy: `random` or `semantic` (requires embeddings).
*   `--base-model`: Target LLM model (overrides config).
*   `--epochs`: Max optimization rounds/candidates.
*   `--demos`: Max number of few-shot examples.
*   `--output`: Output path for the `OptimizedManifest.json`.

### Evaluation

Evaluate an optimized manifest against a test set.

```sh
poetry run coreason-opt evaluate \
    --manifest optimized_manifest.json \
    --dataset data/test_set.csv \
    --metric exact_match
```

**Options:**
*   `--manifest`: Path to the optimized manifest JSON file.
*   `--dataset`: Path to the evaluation dataset.
*   `--metric`: Metric to use: `exact_match` or `f1_score`.

## Configuration

You can configure the optimizer defaults via `OptimizerConfig`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_model` | str | `gpt-4o` | The identifier of the target LLM to optimize for. |
| `meta_model` | str | `gpt-4o` | The identifier of the meta-LLM used for instruction optimization. |
| `metric` | str | `exact_match` | The metric function to use (`exact_match`, `f1_score`). |
| `selector_type` | str | `random` | Strategy for selecting examples (`random`, `semantic`). |
| `embedding_model` | str | `text-embedding-3-small` | Embedding model for semantic selection. |
| `max_bootstrapped_demos` | int | 4 | Maximum number of few-shot examples to bootstrap. |
| `max_rounds` | int | 10 | Maximum number of optimization rounds. |
| `budget_limit_usd` | float | 10.00 | Maximum budget in USD for the run. |

## Development

### Running Tests

To run the test suite with coverage:

```sh
poetry run pytest
```

### Linting and Formatting

This project uses `ruff` and `mypy`.

```sh
# Format
poetry run ruff format .

# Lint
poetry run ruff check --fix .

# Type Check
poetry run mypy .
```

### Pre-commit Hooks

Ensure all checks pass before committing:

```sh
poetry run pre-commit run --all-files
```
