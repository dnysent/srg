"""Weighted sum similarity strategy."""

import numpy as np

from .base import SimilarityStrategy


class WeightedSumStrategy(SimilarityStrategy):
    """Compute individual cosine similarities, then weighted sum."""

    def __init__(self, weights: list[float] | None = None):
        """Initialize with optional weights.

        Args:
            weights: List of weights for each embedding position.
                    If None, uses equal weights (1/N each).
        """
        self.weights = weights

    def _get_weights(self, n_images: int) -> np.ndarray:
        """Get weights array, defaulting to equal weights and normalizing."""
        if self.weights is not None:
            weights = np.array(self.weights[:n_images])
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                return weights / sum_weights
            return weights
        return np.ones(n_images) / n_images

    def compute_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> float:
        """Compute weighted sum of individual cosine similarities.

        Args:
            embeddings_a: Embeddings from first subfolder (N x D).
            embeddings_b: Embeddings from second subfolder (N x D).

        Returns:
            Weighted sum of cosine similarities.
        """
        individual_sims = self.compute_individual_similarities(embeddings_a, embeddings_b)
        weights = self._get_weights(len(individual_sims))
        return float(np.dot(weights, individual_sims))

    def compute_individual_similarities(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> list[float]:
        """Compute individual cosine similarities for corresponding pairs.

        Args:
            embeddings_a: (N x D) array.
            embeddings_b: (N x D) array.

        Returns:
            List of N cosine similarities.
        """
        n_images = embeddings_a.shape[0]
        similarities = []
        for i in range(n_images):
            sim = self.cosine_similarity(embeddings_a[i], embeddings_b[i])
            similarities.append(sim)
        return similarities

    def compute_pairwise_matrix(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings_a: (N x D) array.
            embeddings_b: (M x D) array.

        Returns:
            N x M similarity matrix.
        """
        # Normalize rows
        norm_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        # Handle zero norms
        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)

        a_normalized = embeddings_a / norm_a
        b_normalized = embeddings_b / norm_b

        return a_normalized @ b_normalized.T

    def set_weights(self, weights: list[float]) -> None:
        """Update weights."""
        self.weights = weights
