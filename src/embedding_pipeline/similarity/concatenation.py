"""Concatenation similarity strategy."""

import numpy as np

from .base import SimilarityStrategy


class ConcatenationStrategy(SimilarityStrategy):
    """Concatenate all embeddings into single vector, then compute cosine similarity."""

    def compute_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity of concatenated embedding vectors.

        Args:
            embeddings_a: Embeddings from first subfolder (N x D).
            embeddings_b: Embeddings from second subfolder (N x D).

        Returns:
            Cosine similarity of flattened vectors.
        """
        vec_a = embeddings_a.flatten()
        vec_b = embeddings_b.flatten()
        return self.cosine_similarity(vec_a, vec_b)

    def compute_individual_similarities(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> list[float]:
        """Compute individual cosine similarities for corresponding pairs.

        Note: This method still computes per-position similarities for display,
        even though the main similarity uses concatenation.
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
