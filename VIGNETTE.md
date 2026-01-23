# The Architecture and Utility of coreason-optimizer

## 1. The Philosophy (The Why)

The prevailing method of interacting with Large Language Models (LLMs)—manual "prompt engineering"—is an exercise in frustration. It is artisan work: fragile, unscalable, and often relying on "magic words" that break when models update. The author of `coreason-optimizer` recognizes that prompts are not merely text; they are **trainable parameters** of a software system.

This package exists to replace intuition with optimization. Instead of a developer guessing which few-shot examples might help, `coreason-optimizer` empirically selects them. Instead of rewriting instructions hoping for better JSON compliance, it uses a meta-learner to rewrite them for you. It shifts the paradigm from "Prompt Whisperer" to "Prompt Compiler," treating the agent definition as source code and the deployed prompt as a compiled, frozen binary.

## 2. Under the Hood (The Dependencies & Logic)

The engine runs on a focused stack designed for iterative evaluation:

*   **Pydantic** enforces the rigorous schema definitions (`OptimizerConfig`, `OptimizedManifest`) required for a compiler that must output deterministic artifacts.
*   **OpenAI** & **Numpy/Scikit-Learn** power the semantic search and generation capabilities. The package doesn't just call LLMs; it uses embeddings to find "nearest neighbor" successful examples to inject into prompts (`SemanticSelector`).
*   **Loguru** provides the observability backbone. When an optimization run takes 4 hours and spends $10, you need structured, searchable logs to understand *why* a specific mutation was rejected.
*   **Click** exposes the compiler interface to CI/CD pipelines, allowing optimization to be a step in the build process, not a manual task.

The core logic revolves around the **Mutate-Evaluate Loop**. Inspired by DSPy, the `MiproOptimizer` (Multi-prompt Instruction PRoposal Optimizer) generates candidate instructions using a "Teacher" model. Simultaneously, it selects sets of few-shot examples. It then performs a grid search across these combinations, scoring them against a ground-truth dataset using a defined `Metric` (like `exact_match`). The result is not just a better prompt, but a mathematically optimal one for that specific dataset and model.

## 3. In Practice (The How)

Here is how `coreason-optimizer` transforms a raw agent definition into a deployed artifact.

### Compiling an Agent

The `compile` method is the heart of the system. It takes your agent logic and training data, runs the optimization strategies (like BootstrapFewShot or MIPRO), and returns a frozen manifest.

```python
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.strategies.mipro import MiproOptimizer
from coreason_optimizer.core.metrics import MetricFactory

# 1. Configuration: Define the target environment
config = OptimizerConfig(
    target_model="gpt-4o",
    budget_limit_usd=5.00,  # Safety first
    max_rounds=10,
)

# 2. Instantiate the Optimizer with a specific Metric
# "exact_match" ensures the output strictly adheres to the reference
optimizer = MiproOptimizer(
    llm_client=client, metric=MetricFactory.get("exact_match"), config=config
)

# 3. The Compilation Step
# This runs the "Mutate-Evaluate" loop, finding the best instruction/example pair
manifest = optimizer.compile(
    agent=my_agent_construct,
    trainset=training_examples,
    valset=validation_examples,
)

print(f"Optimization improved score to: {manifest.performance_metric}")
```

### The Optimized Artifact

The output is a portable JSON manifest. This file allows the runtime to execute the optimized agent without needing the optimizer or the training data again.

```python
# The manifest contains the "compiled" prompt logic
print(manifest.optimized_instruction)
# > "Extract adverse events from the text. Format as JSON. [Optimized Instructions...]"

# It also holds the mathematically selected few-shot examples
for example in manifest.few_shot_examples:
    print(f"Input: {example.inputs} -> Output: {example.reference}")
```
