# coreason-optimizer

**The Compiler for the CoReason Agentic Platform.**

[![Organization](https://img.shields.io/badge/org-CoReason--AI-blue)](https://github.com/CoReason-AI)
[![License](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI](https://github.com/CoReason-AI/coreason_optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_optimizer/actions)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**coreason-optimizer** automates the process of prompt engineering by treating prompts as trainable weights. It takes a "Draft Agent" and iterates on it against a ground-truth dataset to mathematically maximize performance metrics, producing a frozen, production-ready manifest.

## Features

*   **Automated Optimization:** Replaces manual prompt tweaking with algorithmic optimization strategies (BootstrapFewShot, MIPRO).
*   **Model-Specific Compilation:** Tunes prompts specifically for the target model (e.g., GPT-4o, Claude 3.5), handling model quirks automatically.
*   **Continuous Learning:**  Easily re-optimize agents as new data becomes available in `coreason-archive`, preventing drift.
*   **Cost Awareness:** Built-in budget management to prevent runaway API costs during optimization runs.
*   **Determinism:** Produces immutable, versioned `OptimizedManifest.json` artifacts for GxP compliance and reproducibility.

## Installation

```bash
pip install coreason-optimizer
```

## Usage

### 1. Optimize an Agent

Use the CLI to optimize an agent defined in `src/agents/analyst.py`.

```bash
# Optimize using MIPRO strategy
coreason-opt tune \
    --agent src/agents/analyst.py \
    --dataset data/gold_set.csv \
    --strategy mipro \
    --output dist/analyst_v2.json
```

### 2. Run the Optimized Agent

Load the manifest and use the optimized prompt in your application code.

```python
import json
from coreason_optimizer.core.models import OptimizedManifest
from coreason_optimizer.core.formatter import format_prompt
from coreason_optimizer.core.client import OpenAIClient

# 1. Load the optimized manifest
with open("dist/analyst_v2.json", "r") as f:
    manifest = OptimizedManifest(**json.load(f))

# 2. Prepare user input
user_input = {"question": "What is the capital of France?"}

# 3. Format the prompt using the optimized instruction and examples
prompt = format_prompt(
    system_prompt=manifest.optimized_instruction,
    examples=manifest.few_shot_examples,
    inputs=user_input
)

# 4. Generate response
client = OpenAIClient()
response = client.generate(
    messages=[{"role": "user", "content": prompt}],
    model=manifest.base_model
)

print(response.content)
```

## Documentation

For more detailed documentation on strategies, configuration, and API references, please refer to the [docs/](docs/) directory or visit our [GitHub pages](https://github.com/CoReason-AI/coreason_optimizer).

## License

This software is proprietary and dual-licensed under the **Prosperity Public License 3.0**.
Commercial use beyond a 30-day trial requires a separate license. See [LICENSE](LICENSE) for details.
