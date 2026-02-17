"""Similarity computer for NxN matrix computation."""

from typing import Callable

import numpy as np
from tqdm import tqdm

from .base import SimilarityStrategy
from .concatenation import ConcatenationStrategy
from .weighted_sum import WeightedSumStrategy
from .svm_strategy import SVMStrategy


class SimilarityComputer:
    """Computes NxN similarity matrix between subfolders."""

    def __init__(self, strategy: SimilarityStrategy):
        """Initialize with similarity strategy.

        Args:
            strategy: The strategy to use for computing similarities.
        """
        self.strategy = strategy

    @classmethod
    def from_method(
        cls,
        method: str,
        weights: list[float] | None = None,
        model_path: str | None = None,
    ) -> "SimilarityComputer":
        """Create SimilarityComputer from method name.

        Args:
            method: Either 'concatenation', 'weighted_sum' or 'svm'.
            weights: Weights for weighted_sum method.
            model_path: Path to pre-trained model for svm method.

        Returns:
            Configured SimilarityComputer.
        """
        if method == "concatenation":
            strategy = ConcatenationStrategy()
        elif method == "weighted_sum":
            strategy = WeightedSumStrategy(weights=weights)
        elif method == "svm":
            if not model_path:
                raise ValueError("Method 'svm' requires a model_path")
            strategy = SVMStrategy(model_path=model_path)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: concatenation, weighted_sum, svm")
        return cls(strategy)


    @staticmethod
    def transform_similarities(
        values: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Apply threshold transformation to similarity values.
        
        Transforms similarities by:
        1. Subtracting threshold
        2. Clamping to minimum 0.0
        3. Rescaling [threshold, 1.0] to [0.0, 1.0]
        
        Formula: max(0, (value - threshold)) / (1 - threshold)
        
        Args:
            values: Similarity values (array or scalar).
            threshold: Threshold to subtract (default 0.5).
            
        Returns:
            Transformed similarity values in [0.0, 1.0].
        """
        if threshold >= 1.0:
            raise ValueError("Threshold must be less than 1.0")
        transformed = np.maximum(0, values - threshold) / (1 - threshold)
        return transformed

    def compute_similarity_matrix(
        self,
        embeddings_dict: dict[str, np.ndarray],
        progress_callback: Callable[[int, int], None] | None = None,
        show_progress: bool = False,
        transform_threshold: float | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute NxN similarity matrix between all subfolders.

        Args:
            embeddings_dict: {subfolder_name: embeddings array (N x D)}.
            progress_callback: Optional callback(current, total) for progress.
            show_progress: Whether to show tqdm progress bar.
            transform_threshold: If set, apply threshold transformation.

        Returns:
            Tuple of (NxN similarity matrix, list of subfolder names).
        """
        names = sorted(embeddings_dict.keys())
        n = len(names)
        matrix = np.zeros((n, n))

        total = n * n
        current = 0

        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing similarity matrix")

        for i in iterator:
            for j in range(n):
                matrix[i, j] = self.strategy.compute_similarity(
                    embeddings_dict[names[i]],
                    embeddings_dict[names[j]],
                )
                current += 1
                if progress_callback:
                    progress_callback(current, total)

        # Apply threshold transformation if requested
        if transform_threshold is not None:
            matrix = self.transform_similarities(matrix, transform_threshold)

        return matrix, names

    def compare_two_subfolders(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        transform_threshold: float | None = None,
    ) -> dict:
        """Compare two subfolders and return detailed comparison.

        Args:
            embeddings_a: First subfolder embeddings (N x D).
            embeddings_b: Second subfolder embeddings (N x D).
            transform_threshold: If set, apply threshold transformation.

        Returns:
            Dict with 'overall_similarity', 'individual_similarities', 'pairwise_matrix'.
        """
        overall = self.strategy.compute_similarity(embeddings_a, embeddings_b)
        individual = self.strategy.compute_individual_similarities(embeddings_a, embeddings_b)
        pairwise = self.strategy.compute_pairwise_matrix(embeddings_a, embeddings_b)
        
        # Apply threshold transformation if requested
        if transform_threshold is not None:
            overall = float(self.transform_similarities(np.array([overall]), transform_threshold)[0])
            individual = self.transform_similarities(np.array(individual), transform_threshold).tolist()
            pairwise = self.transform_similarities(pairwise, transform_threshold)
        
        return {
            "overall_similarity": overall,
            "individual_similarities": individual,
            "pairwise_matrix": pairwise.tolist() if isinstance(pairwise, np.ndarray) else pairwise,
        }

    def set_weights(self, weights: list[float]) -> None:
        """Update weights if using weighted_sum strategy."""
        if isinstance(self.strategy, WeightedSumStrategy):
            self.strategy.set_weights(weights)

