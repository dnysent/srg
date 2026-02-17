"""t-SNE computation for embedding visualization."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN

from ..config import TSNEConfig


@dataclass
class TSNEResult:
    """Result of t-SNE computation."""
    
    coordinates: np.ndarray  # (N, n_components) array
    labels: list[str]  # Label for each point
    subfolder_indices: list[int]  # Which subfolder each point belongs to
    image_indices: list[int]  # Which image index within subfolder
    subfolders: list[str]  # Ordered list of subfolder names
    cluster_labels: Optional[list[int]] = field(default=None)  # Cluster assignments


class TSNEComputer:
    """Computes t-SNE on embedding data."""

    def __init__(self, config: Optional[TSNEConfig] = None):
        """Initialize with configuration.
        
        Args:
            config: t-SNE configuration. If None, use defaults.
        """
        self.config = config or TSNEConfig()

    @staticmethod
    def transform_embeddings(
        embeddings: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Apply threshold transformation to embedding values.
        
        Formula: max(0, (value - threshold)) / (1 - threshold)
        
        Args:
            embeddings: Embedding array to transform.
            threshold: Threshold value (must be < 1.0).
            
        Returns:
            Transformed embeddings.
        """
        if threshold >= 1.0:
            raise ValueError("Threshold must be less than 1.0")
        transformed = np.maximum(0, embeddings - threshold) / (1 - threshold)
        return transformed

    def compute_tsne(
        self,
        embeddings: dict[str, np.ndarray],
        selected_indices: Optional[list[int]] = None,
        scale_factor: float = 1.0,
        transform_threshold: Optional[float] = None,
        cluster_method: Optional[str] = None,
        cluster_k: int = 3,
        cluster_eps: float = 0.5,
        cluster_min_samples: int = 2,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> TSNEResult:
        """Compute t-SNE on all embeddings.
        
        Args:
            embeddings: Dict mapping subfolder names to embedding arrays (N x D).
            selected_indices: Optional list of embedding row indices to use.
                If None, all embeddings are used. Indices are always sorted
                to ensure consistent concatenation order across datapoints.
            scale_factor: Factor to scale embedding vectors by (default 1.0).
            transform_threshold: Optional threshold for transformation.
                Applies max(0, (value - threshold)) / (1 - threshold) to each value.
            cluster_method: Clustering algorithm: "kmeans" or "dbscan" or None.
            cluster_k: Number of clusters for K-Means (default 3).
            cluster_eps: Epsilon for DBSCAN (default 0.5).
            cluster_min_samples: Minimum samples for DBSCAN (default 2).
            progress_callback: Optional callback with (current, total, message).
            
        Returns:
            TSNEResult with coordinates, metadata, and optional cluster labels.
        """
        if progress_callback:
            progress_callback(0, 4, "Preparing embeddings...")
        
        # Collect all embeddings with metadata
        all_embeddings = []
        labels = []
        subfolder_indices = []
        image_indices = []
        subfolders = sorted(embeddings.keys())
        
        # Determine which indices to use (sorted for consistent order)
        first_emb = next(iter(embeddings.values()))
        n_images = first_emb.shape[0]
        
        if selected_indices is not None:
            # Sort indices to ensure consistent concatenation order
            indices_to_use = sorted(selected_indices)
        else:
            indices_to_use = list(range(n_images))
        
        for sf_idx, subfolder in enumerate(subfolders):
            emb = embeddings[subfolder]
            
            # Select only specified embedding rows and concatenate
            selected_embs = emb[indices_to_use]
            
            # Apply threshold transformation if specified
            if transform_threshold is not None:
                selected_embs = self.transform_embeddings(selected_embs, transform_threshold)
            
            # Flatten selected embeddings into single vector per subfolder
            flattened = selected_embs.flatten()
            
            # Apply scaling if not 1.0
            if scale_factor != 1.0:
                flattened = flattened * scale_factor
            
            all_embeddings.append(flattened)
            labels.append(subfolder)
            subfolder_indices.append(sf_idx)
            image_indices.append(-1)  # -1 indicates subfolder-level embedding
        
        X = np.array(all_embeddings)
        
        # Validate embeddings have variance (prevent sklearn crash)
        std = np.std(X)
        if std < 1e-10:
            raise ValueError(
                "Embeddings have near-zero variance. This can happen when threshold "
                "transform removes too much signal. Try a lower threshold value."
            )
        
        if progress_callback:
            progress_callback(1, 4, "Computing t-SNE...")
        
        # Compute t-SNE
        tsne = TSNE(
            n_components=self.config.n_components,
            perplexity=min(self.config.perplexity, len(X) - 1),  # Perplexity must be < n_samples
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        
        coordinates = tsne.fit_transform(X)
        
        if progress_callback:
            progress_callback(2, 4, "Clustering...")
        
        # Perform clustering if requested
        cluster_labels = None
        if cluster_method == "kmeans":
            # K-Means clustering on t-SNE coordinates
            k = min(cluster_k, len(coordinates))  # Can't have more clusters than points
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates).tolist()
        elif cluster_method == "dbscan":
            # DBSCAN clustering on t-SNE coordinates
            dbscan = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
            cluster_labels = dbscan.fit_predict(coordinates).tolist()
        
        if progress_callback:
            progress_callback(3, 4, "Finalizing...")
        
        result = TSNEResult(
            coordinates=coordinates,
            labels=labels,
            subfolder_indices=subfolder_indices,
            image_indices=image_indices,
            subfolders=subfolders,
            cluster_labels=cluster_labels,
        )
        
        if progress_callback:
            progress_callback(4, 4, "Done")
        
        return result
