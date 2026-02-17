"""SVM-based similarity strategy.

Uses a pre-trained Support Vector Regressor to predict similarity
between two sets of embeddings.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .base import SimilarityStrategy
from ..optimization.svm_trainer import SVMTrainer


class SVMStrategy(SimilarityStrategy):
    """SVM-based similarity computation.
    
    Uses a trained SVR model to predict similarity between embedding pairs.
    The model expects concatenated, flattened embeddings as input.
    """

    def __init__(self, model_path: str | Path):
        """Initialize with a pre-trained model.
        
        Args:
            model_path: Path to the trained SVR model file (.pkl)
        """
        self.model_path = Path(model_path)
        self.trainer = SVMTrainer()
        self._model = None

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            self._model = self.trainer.load_model(self.model_path)
        return self._model

    def compute_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> float:
        """Compute similarity between two embedding matrices using the SVM model.
        
        Args:
            embeddings_a: First embedding matrix (n_images, embedding_dim)
            embeddings_b: Second embedding matrix (n_images, embedding_dim)
            
        Returns:
            Predicted similarity score between 0 and 1
        """
        return self.trainer.predict(self.model, embeddings_a, embeddings_b)

    def compute_individual_similarities(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> np.ndarray:
        """Compute per-position cosine similarities (same as weighted_sum).
        
        For SVM strategy, we still compute individual cosine similarities
        for display purposes, though the overall similarity is from SVM.
        
        Args:
            embeddings_a: First embedding matrix (n_images, embedding_dim)
            embeddings_b: Second embedding matrix (n_images, embedding_dim)
            
        Returns:
            Array of individual cosine similarities for each position
        """
        n_images = embeddings_a.shape[0]
        similarities = np.zeros(n_images)
        
        for i in range(n_images):
            a = embeddings_a[i]
            b = embeddings_b[i]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a > 0 and norm_b > 0:
                similarities[i] = np.dot(a, b) / (norm_a * norm_b)
            else:
                similarities[i] = 0.0
        
        return similarities

    def compute_pairwise_matrix(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between embeddings.
        
        For SVM strategy, this computes cosine similarity between each pair
        of individual embeddings (same as other strategies for visualization).
        
        Args:
            embeddings_a: First embedding matrix (n_images, embedding_dim)
            embeddings_b: Second embedding matrix (n_images, embedding_dim)
            
        Returns:
            n_images x n_images pairwise similarity matrix
        """
        n_a = embeddings_a.shape[0]
        n_b = embeddings_b.shape[0]
        matrix = np.zeros((n_a, n_b))
        
        for i in range(n_a):
            for j in range(n_b):
                a = embeddings_a[i]
                b = embeddings_b[j]
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                
                if norm_a > 0 and norm_b > 0:
                    matrix[i, j] = np.dot(a, b) / (norm_a * norm_b)
        
        return matrix
