# Product Requirements Document: coreason-optimizer

**Domain:** Automated Prompt Engineering / LLM Compilation / DSPy Integration
**Package Name:** coreason-optimizer

---

## 1. Executive Summary

**coreason-optimizer** is the "Compiler" for the CoReason Agentic Platform.

In the current SOTA (State-of-the-Art), writing static prompts by hand is considered technical debt. **coreason-optimizer** automates this by treating prompts (instructions and few-shot examples) as **trainable weights**. It ingests a "Draft Agent" defined in `coreason-construct` and iterates on it against a ground-truth dataset (validated by `coreason-assay`), mathematically maximizing performance metrics. It outputs a "Frozen Manifest" that is deployed to production, ensuring GxP stability.

## 2. Problem Statement & Rationale

| Problem | Impact | The coreason-optimizer Solution |
| :---- | :---- | :---- |
| **The "Prompt Whisperer" Bottleneck** | Engineers spend hours tweaking words ("Please be careful") with unpredictable results. | **Automated Optimization:** A meta-algorithm rewrites instructions and selects examples to maximize a score, not human intuition. |
| **Brittleness** | A prompt that works for GPT-4 often fails for Claude 3.5 or Llama 3. | **Model-Specific Compilation:** The optimizer can run separate jobs to generate optimized prompts specifically tuned for the target model. |
| **Drift** | Agents degrade over time as data distributions change (e.g., new medical slang). | **Continuous Learning:** Re-running the optimizer on recent "Gold" logs from `coreason-archive` automatically patches the prompt. |

## 3. Architectural Design

### 3.1 The "Mutate-Evaluate" Loop

The package implements a systematic optimization cycle (inspired by DSPy):

1.  **Draft:** Start with the developer's base intention.
2.  **Evaluate:** Run the agent on a training set.
3.  **Diagnose:** Identify failing examples using `coreason-assay` metrics.
4.  **Mutate:**
    *   **Bootstrap Few-Shot:** Find historical examples where the agent *succeeded* on similar hard cases and inject them into the prompt.
    *   **Instruction Induction:** Use a Meta-LLM to rewrite the System Prompt to explicitly address the observed failures.
5.  **Select:** Keep the mutation that yields the highest metric score.

### 3.2 Integration Map

*   **Input (Schema):** `coreason-construct` defines the Agent structure (Inputs/Outputs).
*   **Input (Data):** `coreason-archive` provides historical logs to mine for training examples.
*   **Feedback (Loss Function):** `coreason-assay` provides the scoring function (e.g., accuracy, json_validity, f1_score).
*   **Output (Artifact):** Produces a versioned `OptimizedManifest.json` used by the runtime.

## 4. Functional Specifications

### 4.1 The Optimization Engine

*   **Strategy: BootstrapFewShot:**
    *   Automatically mines the "Teacher" model's successful traces to create few-shot examples for the "Student" prompt.
*   **Strategy: MIPRO (Multi-prompt Instruction PRoposal Optimizer):**
    *   Generates 10 candidates for the System Instruction and 5 combinations of Few-Shot examples, finding the optimal pair via Bayesian optimization or simple grid search.
*   **Cost Awareness:**
    *   Must implement a `BudgetManager` to halt optimization if the token spend exceeds a defined limit (e.g., $10.00).

### 4.2 Data Management

*   **Dataset Loader:** Standardizes inputs from CSV, JSONL, or `coreason-archive` SQL queries into a `TrainingExample` object.
*   **Splitter:** automatically creates Train/Dev/Test splits to prevent overfitting the prompt to the training data.

### 4.3 The Manifest Serializer

*   The output must be deterministic and immutable.
*   **Schema:**
    ```json
    {
      "agent_id": "adverse_event_extractor",
      "base_model": "gpt-4o",
      "optimized_instruction": "Extract adverse events... [Modified by Optimizer]",
      "few_shot_examples": [ ... ],
      "performance_metric": "0.94",
      "optimization_run_id": "opt_20250119_xyz"
    }
    ```

## 5. Technical Specifications (API)

### 5.1 The Interface

```python
class OptimizerConfig(BaseModel):
    target_model: str = "gpt-4o"
    metric: str = "exact_match"
    max_bootstrapped_demos: int = 4
    max_rounds: int = 10

class PromptOptimizer(ABC):
    @abstractmethod
    def compile(self,
                agent: Construct,
                trainset: List[Example],
                valset: List[Example]) -> OptimizedManifest:
        """Run the optimization loop."""
        pass
```

### 5.2 The CLI (coreason-opt)

The package should expose a command-line interface for CI/CD integration:

*   `coreason-opt tune --agent src/agents/analyst.py --dataset data/gold_set.csv`
*   `coreason-opt evaluate --manifest dist/analyst_v2.json --dataset data/test_set.csv`

## 6. Implementation Plan: Atomic Units of Change (AUC)

### Phase 1: Foundation

*   **AUC-1: Scaffold & Configuration:** Project structure, `pyproject.toml`, and `OptimizerConfig` Pydantic models.
*   **AUC-2: Abstract Base Classes:** Define `BaseOptimizer`, `BaseSelector` (for examples), and `BaseMutator` (for instructions).

### Phase 2: Data & Metrics

*   **AUC-3: Dataset Loader:** Implement `Dataset` class that handles loading/splitting from CSV and `coreason-archive`.
*   **AUC-4: Metric Adapter:** Create a wrapper that adapts `coreason-assay` functions into the format required by the optimization loop.

### Phase 3: The Strategies

*   **AUC-5: Few-Shot Selector:** Implement logic to select examples using Semantic Similarity (via `coreason-foundry` embeddings) or Random Sampling.
*   **AUC-6: Bootstrap Logic:** Implement the "Teacher-Student" loop where the model generates its own training data from input questions.
*   **AUC-7: Instruction Mutator:** Implement the Meta-Prompt that analyzes failures and rewrites the system prompt.

### Phase 4: The Loop & Artifacts

*   **AUC-8: The Compile Loop:** Connect the Mutators and Selectors into the main `compile()` orchestration method.
*   **AUC-9: Manifest Serializer:** Logic to dump the final state to JSON.
*   **AUC-10: CLI Entrypoint:** Build the `coreason-opt` command line tool.

## 7. Compliance & Safety

*   **Audit Trail:** Every optimization run must log the `trace_id` of the experiments to `coreason-veritas`. We must be able to explain *why* the prompt changed.
*   **Human-in-the-Loop Gate:** The `OptimizedManifest` is not automatically deployed. It is saved as a "Candidate" that requires a human to review the score improvement before promotion to production.
