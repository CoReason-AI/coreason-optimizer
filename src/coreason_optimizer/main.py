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
CLI Entrypoint for the Coreason Optimizer.

This module provides the command-line interface for the optimization tool,
supporting commands to tune agents and evaluate manifests.
"""

import json
from pathlib import Path

import click
from coreason_identity.models import UserContext

from coreason_optimizer.core.client import OpenAIClient, OpenAIEmbeddingClient, OptimizationClient
from coreason_optimizer.core.config import OptimizerConfig
from coreason_optimizer.core.formatter import format_prompt
from coreason_optimizer.core.interfaces import PromptOptimizer
from coreason_optimizer.core.metrics import MetricFactory
from coreason_optimizer.core.models import OptimizedManifest
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.strategies.bootstrap import BootstrapFewShot
from coreason_optimizer.strategies.mipro import MiproOptimizer
from coreason_optimizer.strategies.selector import StrategySelector
from coreason_optimizer.utils.import_utils import load_agent_from_path
from coreason_optimizer.utils.logger import logger


@click.group()
def cli() -> None:
    """coreason-opt: The Compiler for the CoReason Agentic Platform."""
    pass


@cli.command()
@click.option(
    "--agent",
    required=True,
    help="Path to the agent file (e.g., src/agents/analyst.py[:var])",
)
@click.option("--dataset", required=True, help="Path to the dataset (CSV or JSONL)")
@click.option("--base-model", help="Target LLM model (overrides config)")
@click.option("--epochs", type=int, help="Max optimization rounds (overrides config)")
@click.option("--demos", type=int, help="Max bootstrapped demos (overrides config)")
@click.option(
    "--output",
    default="optimized_manifest.json",
    help="Output path for the manifest",
)
@click.option(
    "--strategy",
    type=click.Choice(["mipro", "bootstrap"]),
    default="mipro",
    help="Optimization strategy",
)
@click.option(
    "--selector",
    type=click.Choice(["random", "semantic"]),
    help="Selector strategy (random or semantic)",
)
def tune(
    agent: str,
    dataset: str,
    base_model: str | None,
    epochs: int | None,
    demos: int | None,
    output: str,
    strategy: str,
    selector: str | None,
) -> None:
    """
    Optimize an agent's prompt against a dataset.

    Args:
        agent: Path to the agent file (and optional variable name).
        dataset: Path to the dataset file (.csv or .jsonl).
        base_model: Target LLM model identifier.
        epochs: Maximum number of optimization rounds.
        demos: Maximum number of few-shot examples to include.
        output: Path to save the resulting OptimizedManifest JSON.
        strategy: Optimization strategy to use ('mipro' or 'bootstrap').
        selector: Few-shot example selection strategy ('random' or 'semantic').
    """
    # Create System Context
    system_context = UserContext(
        user_id="cli-user",
        email="cli-user@coreason.ai",
        groups=["system"],
        claims={"source": "cli"},
    )

    logger.info(
        f"Starting optimization for agent: {agent}",
        user_id=system_context.user_id,
    )

    # Initialize Optimization Client (Audit)
    opt_client = OptimizationClient()
    # We store the study_id but don't strictly use it yet in this version of the optimizer,
    # but we register it for audit compliance.
    _ = opt_client.register_study(f"opt-{Path(agent).stem}", context=system_context)

    # Validate Strategy
    strat_selector = StrategySelector()
    strategy = strat_selector.select_strategy(strategy, context=system_context)

    # Load Agent
    try:
        construct = load_agent_from_path(agent)
    except Exception as e:
        logger.error(f"Failed to load agent: {e}")
        raise click.ClickException(str(e)) from e

    # Load Dataset
    try:
        ds_path = Path(dataset)
        if ds_path.suffix.lower() == ".jsonl":
            full_ds = Dataset.from_jsonl(ds_path, context=system_context)
        elif ds_path.suffix.lower() == ".csv":
            # Assume reference col is 'reference' and inputs are from construct
            input_cols = construct.inputs
            full_ds = Dataset.from_csv(
                ds_path, input_cols=input_cols, reference_col="reference", context=system_context
            )
        else:
            raise click.ClickException("Unsupported file format. Use .csv or .jsonl")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")  # pragma: no cover
        raise click.ClickException(str(e)) from e  # pragma: no cover

    # Split Data (simple train/val)
    train_set, val_set, _ = full_ds.split(train_ratio=0.8, val_ratio=0.2)
    # Convert Dataset back to list[TrainingExample] as required by compile
    train_list = list(train_set)
    val_list = list(val_set)

    # Config
    config = OptimizerConfig()  # Defaults
    if base_model:
        config.target_model = base_model
    if epochs:
        config.max_rounds = epochs
    if demos:
        config.max_bootstrapped_demos = demos
    if selector:
        config.selector_type = selector  # type: ignore

    # Client
    # Uses OPENAI_API_KEY env var
    try:
        client = OpenAIClient()
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Client: {e}")  # pragma: no cover
        raise click.ClickException(
            "Failed to initialize OpenAI Client. Check OPENAI_API_KEY."
        ) from e  # pragma: no cover

    # Metric
    try:
        metric = MetricFactory.get(config.metric)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    # Optimizer
    optimizer: PromptOptimizer
    if strategy == "bootstrap":
        optimizer = BootstrapFewShot(client, metric, config)
    else:
        embedding_provider = None
        if config.selector_type == "semantic":
            # Initialize embedding provider
            try:
                embedding_provider = OpenAIEmbeddingClient()
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI Embedding Client: {e}")  # pragma: no cover
                raise click.ClickException(
                    "Failed to initialize OpenAI Embedding Client. Check OPENAI_API_KEY."
                ) from e  # pragma: no cover

        optimizer = MiproOptimizer(client, metric, config, embedding_provider=embedding_provider)

    # Run
    try:
        manifest = optimizer.compile(construct, train_list, val_list)
    except Exception as e:
        logger.exception("Optimization failed")  # pragma: no cover
        raise click.ClickException(f"Optimization failed: {e}") from e  # pragma: no cover

    # Save
    try:
        with open(output, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        logger.info(f"Manifest saved to {output}")
        click.echo(f"Optimization complete. Score: {manifest.performance_metric:.4f}. Manifest saved to {output}")
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")  # pragma: no cover
        raise click.ClickException(str(e)) from e  # pragma: no cover


@cli.command()
@click.option("--manifest", required=True, help="Path to the optimized manifest JSON")
@click.option("--dataset", required=True, help="Path to the evaluation dataset")
@click.option("--metric", default="exact_match", help="Metric to use for evaluation")
def evaluate(manifest: str, dataset: str, metric: str) -> None:
    """
    Evaluate an optimized manifest against a dataset.

    Args:
        manifest: Path to the optimized manifest JSON file.
        dataset: Path to the evaluation dataset file.
        metric: The metric to use for scoring (e.g., 'exact_match').
    """
    # Load Manifest
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest_obj = OptimizedManifest(**data)
    except Exception as e:
        raise click.ClickException(f"Failed to load manifest: {e}") from e

    # Create System Context for Evaluation
    system_context = UserContext(
        user_id="cli-evaluator",
        email="evaluator@coreason.ai",
        groups=["system"],
        claims={"source": "cli-eval"},
    )

    # Load Dataset
    try:
        ds_path = Path(dataset)
        if ds_path.suffix.lower() == ".jsonl":
            eval_ds = Dataset.from_jsonl(ds_path, context=system_context)
        else:
            # Fallback for CSV: try to use keys from first few-shot example if available
            if manifest_obj.few_shot_examples:
                input_cols = list(manifest_obj.few_shot_examples[0].inputs.keys())
                eval_ds = Dataset.from_csv(
                    ds_path,
                    input_cols=input_cols,
                    reference_col="reference",
                    context=system_context,
                )
            else:
                raise click.ClickException(
                    "Cannot infer CSV schema for evaluation without few-shot examples in manifest. Use JSONL."
                )
    except Exception as e:
        raise click.ClickException(f"Failed to load dataset: {e}") from e

    # Setup Evaluation
    try:
        client = OpenAIClient()
    except Exception:
        raise click.ClickException(
            "Failed to initialize OpenAI Client. Check OPENAI_API_KEY."
        ) from None  # pragma: no cover

    try:
        metric_func = MetricFactory.get(metric)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    total_score = 0.0
    count = 0

    logger.info("Starting evaluation...")
    # Convert to list to iterate with progress bar
    examples_list = list(eval_ds)

    with click.progressbar(examples_list, label="Evaluating") as bar:
        for example in bar:
            # Reconstruct prompt using optimized instruction and examples
            prompt = format_prompt(
                system_prompt=manifest_obj.optimized_instruction,
                examples=manifest_obj.few_shot_examples,
                inputs=example.inputs,
            )
            try:
                response = client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=manifest_obj.base_model,
                    temperature=0.0,
                )
                score = metric_func(response.content, example.reference)
                total_score += score
                count += 1
            except Exception as e:
                logger.warning(f"Error evaluating example: {e}")

    avg_score = total_score / count if count > 0 else 0.0
    click.echo(f"Evaluation Complete. Average {metric} Score: {avg_score:.4f}")


if __name__ == "__main__":  # pragma: no cover
    cli()  # pragma: no cover
