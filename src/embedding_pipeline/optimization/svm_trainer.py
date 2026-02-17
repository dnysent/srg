"""SVM-based similarity training and prediction.

This module provides tools for training and using Support Vector Machines
for predicting similarity between image embedding pairs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..config import SVMConfig
from .losses import get_loss_function

logger = logging.getLogger(__name__)


@dataclass
class SVMTrainingResult:
    """Result of SVM training."""

    model: SVR
    train_mse: float
    train_mae: float
    train_r2: float
    val_mse: float
    val_mae: float
    val_r2: float
    n_train_samples: int
    n_val_samples: int
    train_loss: float = 0.0  # Loss based on config.cost_function
    val_loss: float = 0.0


class SVMTrainer:
    """Trains SVM models for embedding similarity prediction.
    
    The trainer creates training pairs from embeddings and ground-truth matrix,
    then trains a Support Vector Regressor to predict similarity scores.
    """

    def __init__(self, config: Optional[SVMConfig] = None):
        """Initialize SVM trainer.
        
        Args:
            config: SVM training configuration. Uses defaults if None.
        """
        self.config = config or SVMConfig()

    def create_training_data(
        self,
        embeddings: dict[str, np.ndarray],
        ground_truth: np.ndarray,
        gt_labels: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create training pairs from embeddings and ground-truth matrix.
        
        For each pair (i, j) of subfolders, creates a training sample where:
        - X = concatenation of flattened embeddings from subfolder i and j
        - y = ground_truth[i, j] (the known similarity)
        
        If gt_labels is provided, ground_truth[i, j] is assumed to correspond to
        the pair (gt_labels[i], gt_labels[j]).
        
        Args:
            embeddings: Dict mapping subfolder names to embedding matrices (n_images, dim)
            ground_truth: NxN similarity matrix where N = number of subfolders
            gt_labels: Optional list of labels for x/y axis of ground_truth
            
        Returns:
            Tuple of (X, y) where X is (n_pairs, 2*n_images*dim) and y is (n_pairs,)
        """
        if gt_labels:
            subfolder_names = gt_labels
            # Verify all labels exist in embeddings
            for name in subfolder_names:
                if name not in embeddings:
                    raise ValueError(f"Label '{name}' in gt_labels not found in embeddings")
        else:
            subfolder_names = sorted(embeddings.keys())
            
        n_subfolders = len(subfolder_names)
        
        if ground_truth.shape != (n_subfolders, n_subfolders):
            raise ValueError(
                f"Ground truth shape {ground_truth.shape} doesn't match "
                f"number of subfolders ({n_subfolders})"
            )
        
        # Determine features per training sample
        # Each sample is a concatenation of two flattened component matrices
        first_name = subfolder_names[0]
        n_images, dim = embeddings[first_name].shape
        sample_size = 2 * n_images * dim
        
        X = np.zeros((n_subfolders * n_subfolders, sample_size), dtype=np.float32)
        y = np.zeros(n_subfolders * n_subfolders, dtype=np.float32)
        
        idx = 0
        for i, name_i in enumerate(subfolder_names):
            emb_i = embeddings[name_i].flatten()
            for j, name_j in enumerate(subfolder_names):
                emb_j = embeddings[name_j].flatten()
                
                # Check consistency
                if len(emb_i) != n_images * dim or len(emb_j) != n_images * dim:
                    raise ValueError(
                        f"Inconsistent embedding size for {name_i} or {name_j}. "
                        f"Expected {n_images * dim} elements (flattened)."
                    )
                
                X[idx] = np.concatenate([emb_i, emb_j])
                y[idx] = ground_truth[i, j]
                idx += 1
        
        logger.info(f"Created {len(X)} training pairs from {n_subfolders} subfolders. "
                    f"Images per component: {n_images}")
        
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SVMTrainingResult:
        """Train the SVM model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            progress_callback: Optional callback for progress updates
            
        Returns:
            SVMTrainingResult with trained model and metrics
        """
        if progress_callback:
            progress_callback("Splitting data into train/validation sets...")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        
        logger.info(f"Train set: {len(X_train)} samples, Val set: {len(X_val)} samples")
        
        if progress_callback:
            progress_callback(f"Training SVR with {len(X_train)} samples...")
        
        # Create SVR model
        # Handle gamma - can be 'scale', 'auto', or a float
        gamma = self.config.gamma
        if isinstance(gamma, str) and gamma not in ('scale', 'auto'):
            try:
                gamma = float(gamma)
            except ValueError:
                gamma = 'scale'
        
        model = SVR(
            kernel=self.config.kernel,
            C=self.config.C,
            gamma=gamma,
            epsilon=self.config.epsilon,
            degree=self.config.degree,
        )
        
        # Train
        model.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback("Evaluating model...")
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Compute custom loss
        loss_fn = get_loss_function(self.config.cost_function)
        if self.config.cost_function == "thresholded":
            train_loss = loss_fn(y_train_pred, y_train, threshold=self.config.cost_threshold)
            val_loss = loss_fn(y_val_pred, y_val, threshold=self.config.cost_threshold)
        else:
            train_loss = loss_fn(y_train_pred, y_train)
            val_loss = loss_fn(y_val_pred, y_val)

        logger.info(f"Training MSE: {train_mse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")
        logger.info(f"Validation MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
        logger.info(f"Custom Loss ({self.config.cost_function}): Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        return SVMTrainingResult(
            model=model,
            train_mse=train_mse,
            train_mae=train_mae,
            train_r2=train_r2,
            val_mse=val_mse,
            val_mae=val_mae,
            val_r2=val_r2,
            train_loss=train_loss,
            val_loss=val_loss,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val),
        )

    def save_model(self, model: SVR, path: str | Path) -> None:
        """Save trained model to file.
        
        Args:
            model: Trained SVR model
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path) -> SVR:
        """Load trained model from file.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded SVR model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model

    def predict(
        self,
        model: SVR,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> float:
        """Predict similarity between two embedding matrices.
        
        Args:
            model: Trained SVR model
            embeddings_a: Embedding matrix for first subfolder (n_images, dim)
            embeddings_b: Embedding matrix for second subfolder (n_images, dim)
            
        Returns:
            Predicted similarity score, clamped to [0, 1]
        """
        # Concatenate and flatten
        x = np.concatenate([embeddings_a.flatten(), embeddings_b.flatten()])
        x = x.reshape(1, -1)  # Reshape for single prediction
        
        # Predict
        similarity = model.predict(x)[0]
        
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
