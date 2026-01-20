# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_optimizer

import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans

from coreason_optimizer.core.interfaces import EmbeddingProvider
from coreason_optimizer.core.models import TrainingExample
from coreason_optimizer.data.loader import Dataset


class BaseSelector(ABC):
    """Abstract base class for few-shot example selection strategies."""

    @abstractmethod
    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """Select k examples from the training set."""
        pass  # pragma: no cover


class RandomSelector(BaseSelector):
    """Randomly selects examples from the training set."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """Select k random examples."""
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

    def __init__(self, embedding_provider: EmbeddingProvider, seed: int = 42):
        self.embedding_provider = embedding_provider
        self.seed = seed

    def select(self, trainset: Dataset, k: int = 4) -> list[TrainingExample]:
        """Select k diverse examples using clustering."""
        if len(trainset) <= k:
            return list(trainset)

        # 1. Prepare texts for embedding
        texts = []
        for ex in trainset:
            # Simple serialization: "input1: val1\ninput2: val2"
            text = "\n".join(f"{key}: {val}" for key, val in ex.inputs.items())
            texts.append(text)

        # 2. Get embeddings
        embeddings = self.embedding_provider.embed(texts)
        X = np.array(embeddings)

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
