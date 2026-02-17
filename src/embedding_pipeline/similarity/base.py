"""Abstract base class for similarity strategies."""

from abc import ABC, abstractmethod

import numpy as np


class SimilarityStrategy(ABC):
    """Base class for similarity computation strategies."""

    @abstractmethod
    def compute_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> float:
        """Compute overall similarity between two sets of embeddings.

        Args:
            embeddings_a: Embeddings from first subfolder (N x D).
            embeddings_b: Embeddings from second subfolder (N x D).

        Returns:
            Single similarity score.
        """
        pass

    @abstractmethod
    def compute_individual_similarities(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> list[float]:
        """Compute individual similarities for corresponding embeddings.

        Args:
            embeddings_a: Embeddings from first subfolder (N x D).
            embeddings_b: Embeddings from second subfolder (N x D).

        Returns:
            List of N similarity scores (one per corresponding pair).
        """
        pass

    @abstractmethod
    def compute_pairwise_matrix(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between all embeddings.

        Args:
            embeddings_a: Embeddings from first subfolder (N x D).
            embeddings_b: Embeddings from second subfolder (M x D).

        Returns:
            N x M similarity matrix.
        """
        pass

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
