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
Example Selection Strategy.

This module provides classes to select a subset of training examples to be used
as few-shot demonstrations, using either random sampling or semantic clustering.
"""

import json
import random
from abc import ABC, abstractmethod

import numpy as np
from coreason_identity.models import UserContext
from sklearn.cluster import KMeans

from coreason_optimizer.core.interfaces import EmbeddingProvider
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset
from coreason_optimizer.utils.logger import logger


class BaseSelector(ABC):
    """Abstract base class for few-shot example selection strategies."""

    @abstractmethod
    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """
        Select k examples from the training set.

        Args:
            trainset: The source dataset.
            k: The number of examples to select.

        Returns:
            A list of selected TrainingExample objects.
        """
        pass  # pragma: no cover


class RandomSelector(BaseSelector):
    """Randomly selects examples from the training set."""

    def __init__(self, seed: int = 42):
        """
        Initialize RandomSelector.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed

    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """
        Select k random examples.

        Args:
            trainset: The source dataset.
            k: Number of examples to select.

        Returns:
            List of randomly selected examples.
        """
        if len(trainset) <= k:
            return list(trainset)

        rng = random.Random(self.seed)
        return rng.sample(list(trainset), k)


class SemanticSelector(BaseSelector):
    """
    Selects diverse examples using K-Means clustering on embeddings.

    Logic:
    1. Embed all examples.
    2. Cluster into k clusters.
    3. Select the example closest to the centroid of each cluster.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        seed: int = 42,
        embedding_model: str | None = None,
    ):
        """
        Initialize SemanticSelector.

        Args:
            embedding_provider: Provider to generate embeddings.
            seed: Random seed for clustering initialization.
            embedding_model: Optional specific model to use for embeddings.
        """
        self.embedding_provider = embedding_provider
        self.seed = seed
        self.embedding_model = embedding_model

    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """
        Select k diverse examples using clustering.

        Args:
            trainset: The source dataset.
            k: Number of examples to select.

        Returns:
            List of diverse examples.
        """
        if len(trainset) <= k:
            return list(trainset)

        # 1. Prepare texts for embedding
        texts = []
        for ex in trainset:
            # Use JSON serialization for robustness
            text = json.dumps(ex.inputs, sort_keys=True)
            texts.append(text)

        # 2. Get embeddings
        response = self.embedding_provider.embed(texts, model=self.embedding_model)
        X = np.array(response.embeddings)

        # 3. K-Means Clustering
        # n_init="auto" is default in newer sklearn, explicit for safety
        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        kmeans.fit(X)

        # 4. Select representatives (closest to centroid)
        selected_indices = []
        for i in range(k):
            centroid = kmeans.cluster_centers_[i]

            # Find points belonging to this cluster
            cluster_indices = np.where(kmeans.labels_ == i)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_points = X[cluster_indices]
            # Calculate Euclidean distance from centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_idx_in_cluster = np.argmin(distances)
            original_idx = cluster_indices[closest_idx_in_cluster]
            selected_indices.append(original_idx)

        # Handle potential duplicates or fewer points
        selected_indices = sorted(list(set(selected_indices)))

        # Fill if needed
        if len(selected_indices) < k:
            remaining_indices = [idx for idx in range(len(trainset)) if idx not in selected_indices]
            rng = random.Random(self.seed)
            needed = k - len(selected_indices)
            if remaining_indices:
                extra = rng.sample(remaining_indices, min(len(remaining_indices), needed))
                selected_indices.extend(extra)
                selected_indices.sort()

        return [trainset[idx] for idx in selected_indices]


class StrategySelector:
    """Selector for choosing the optimization strategy based on identity and policy."""

    def select_strategy(self, strategy: str, context: UserContext) -> str:
        """
        Select and validate the optimization strategy.

        Args:
            strategy: The requested strategy name.
            context: The user context.

        Returns:
            The authorized strategy name.

        Raises:
            ValueError: If context is missing.
        """
        if context is None:
            raise ValueError("UserContext is required.")

        logger.info(
            "Selecting optimization strategy",
            user_id=context.user_id,
            authorized_strategies=context.claims.get("strategies", "all"),
        )

        return strategy
