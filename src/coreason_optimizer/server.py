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
FastAPI Server implementation for the Coreason Optimization Microservice.
"""

import contextlib
from typing import Any, AsyncIterator, cast

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

from coreason_optimizer.core.client import OpenAIClientAsync, OpenAIEmbeddingClientAsync
from coreason_optimizer.core.interfaces import (
    AsyncEmbeddingProvider,
    AsyncLLMClient,
    EmbeddingResponse,
    LLMResponse,
)
from coreason_optimizer.core.metrics import MetricFactory
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.server_schemas import AgentDefinition, OptimizationRequest
from coreason_optimizer.strategies.bootstrap import BootstrapFewShot
from coreason_optimizer.strategies.mipro import MiproOptimizer

# --- Adapters ---


class DynamicConstruct:
    """
    Adapter to make AgentDefinition satisfy the Construct protocol.
    """

    def __init__(self, agent_def: AgentDefinition):
        self._agent = agent_def

    @property
    def inputs(self) -> list[str]:
        return self._agent.inputs

    @property
    def outputs(self) -> list[str]:
        return self._agent.outputs

    @property
    def system_prompt(self) -> str:
        return self._agent.system_prompt


class BridgedLLMClient:
    """
    Sync wrapper that bridges calls to an AsyncLLMClient running in the main loop.

    This allows synchronous strategies to use async clients managed by FastAPI.
    """

    def __init__(self, async_client: AsyncLLMClient):
        self.async_client = async_client

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        async def _call() -> LLMResponse:
            return await self.async_client.generate(messages=messages, model=model, temperature=temperature, **kwargs)

        # Dispatch to the main event loop
        return cast(LLMResponse, anyio.from_thread.run(_call))


class BridgedEmbeddingProvider:
    """
    Sync wrapper that bridges calls to an AsyncEmbeddingProvider running in the main loop.
    """

    def __init__(self, async_provider: AsyncEmbeddingProvider):
        self.async_provider = async_provider

    def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        async def _call() -> EmbeddingResponse:
            return await self.async_provider.embed(texts=texts, model=model)

        # Dispatch to the main event loop
        return cast(EmbeddingResponse, anyio.from_thread.run(_call))


# --- Lifespan ---


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage the lifecycle of shared clients.
    """
    # Initialize shared clients
    # We use a single httpx client for connection pooling
    http_client = httpx.AsyncClient()

    # Initialize AsyncOpenAI with the shared http client
    # This expects OPENAI_API_KEY to be present in environment variables
    openai_client = AsyncOpenAI(http_client=http_client)

    app.state.http_client = http_client
    app.state.openai_client = openai_client

    # Create Coreason Async wrappers using the shared clients
    # We pass the shared clients so they are reused
    app.state.llm_client_async = OpenAIClientAsync(client=openai_client, http_client=http_client)
    app.state.embedding_client_async = OpenAIEmbeddingClientAsync(client=openai_client, http_client=http_client)

    yield

    # Cleanup
    await openai_client.close()
    await http_client.aclose()


# --- Server ---

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/optimize")
def optimize(request: Request, body: OptimizationRequest) -> Any:
    # 1. Adapter: Convert AgentDefinition to Construct
    agent = DynamicConstruct(body.agent)

    # 2. Dataset: Load and split
    # We use the Dataset class to handle splitting logic
    dataset = Dataset(body.dataset)
    # Default 80/20 split as requested
    train_set, val_set, _ = dataset.split(train_ratio=0.8, val_ratio=0.2)
    train_list = list(train_set)
    val_list = list(val_set)

    # 3. Clients: Bridge to shared async clients
    if not hasattr(request.app.state, "llm_client_async"):
        raise HTTPException(status_code=500, detail="LLM Client not initialized")

    llm_client = BridgedLLMClient(request.app.state.llm_client_async)

    embedding_provider = None
    if body.config.selector_type == "semantic":
        if not hasattr(request.app.state, "embedding_client_async"):
            raise HTTPException(status_code=500, detail="Embedding Client not initialized")
        embedding_provider = BridgedEmbeddingProvider(request.app.state.embedding_client_async)

    # 4. Metric
    try:
        metric = MetricFactory.get(body.config.metric)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # 5. Strategy: Initialize the optimizer
    optimizer: Any
    if body.strategy == "bootstrap":
        optimizer = BootstrapFewShot(llm_client, metric, body.config)
    else:
        # Default to Mipro
        optimizer = MiproOptimizer(llm_client, metric, body.config, embedding_provider=embedding_provider)

    # 6. Run Compilation
    try:
        manifest = optimizer.compile(agent, train_list, val_list)
        return manifest
    except Exception as e:
        # In production, we should log the full traceback
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}") from e
