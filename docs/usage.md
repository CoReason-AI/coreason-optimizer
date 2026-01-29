# Usage Guide

`coreason-optimizer` offers three ways to compile and optimize agents:
1.  **Python Library:** Integrate directly into your Python code.
2.  **CLI:** Use the command-line tool for scripts and CI/CD.
3.  **Server Mode (Microservice):** Run as a standalone API service.

---

## 1. Python Library

Use the library to programmatically compile agents.

```python
from coreason_optimizer import OptimizerConfig, PromptOptimizer
from coreason_optimizer.core.interfaces import Construct
from coreason_optimizer.data import Dataset

# 1. Define your Agent
class MyAgent(Construct):
    inputs = ["user_query"]
    outputs = ["response", "thought_trace"]
    system_prompt = "You are a helpful AI assistant."

agent = MyAgent()

# 2. Load Data
dataset = Dataset.from_csv("data/training_data.csv")
train_set, val_set = dataset.split(train_ratio=0.8)

# 3. Configure Optimizer
config = OptimizerConfig(
    target_model="gpt-4o",
    metric="exact_match",
    budget_limit_usd=5.0
)

# 4. Compile
optimizer = PromptOptimizer(config=config)
manifest = optimizer.compile(agent, train_set, val_set)

print(f"Optimization Score: {manifest.performance_metric}")
```

---

## 2. Command Line Interface (CLI)

The `coreason-opt` CLI allows you to run optimization jobs from the shell.

### Tune an Agent

```bash
coreason-opt tune \
    --agent src/agents/analyst.py \
    --dataset data/gold_set.csv \
    --strategy mipro \
    --output optimized_analyst.json
```

### Evaluate a Manifest

```bash
coreason-opt evaluate \
    --manifest optimized_analyst.json \
    --dataset data/test_set.csv
```

---

## 3. Server Mode (Optimization-as-a-Service)

You can run `coreason-optimizer` as a containerized microservice that accepts optimization requests via HTTP.

### Starting the Server

**Using Docker:**

```bash
docker build -t coreason-optimizer:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... coreason-optimizer:latest
```

**Using Uvicorn (Locally):**

```bash
uvicorn coreason_optimizer.server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### `POST /optimize`

Submits an optimization job.

**Request Body (JSON):**

```json
{
  "agent": {
    "system_prompt": "You are a specialized medical analyst...",
    "inputs": ["patient_notes"],
    "outputs": ["diagnosis_code"]
  },
  "dataset": [
    {
      "inputs": {"patient_notes": "Patient complains of..."},
      "reference": "E11.9",
      "metadata": {"source": "manual_review"}
    },
    ...
  ],
  "config": {
    "target_model": "gpt-4o",
    "metric": "exact_match",
    "budget_limit_usd": 10.0
  },
  "strategy": "mipro"
}
```

**Response:**

Returns an `OptimizedManifest` JSON object containing the optimized instruction and selected few-shot examples.

#### `GET /health`

Checks service health.

**Response:** `{"status": "ready"}`
