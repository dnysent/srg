"""Loss functions for similarity optimization."""

import numpy as np


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return float(np.mean((y_pred - y_true) ** 2))


def thresholded_mse_loss(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    """Thresholded Mean Squared Error loss.
    
    Converts predictions to 1.0 or 0.0 based on threshold before computing MSE.
    Useful when ground truth is binary and we only care about the decision boundary.
    """
    y_pred_bin = (y_pred >= threshold).astype(float)
    return float(np.mean((y_pred_bin - y_true) ** 2))


def binary_cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-15) -> float:
    """Binary Cross-Entropy loss.
    
    Expects y_true to be binary (0 or 1) and y_pred to be in [0, 1].
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def get_loss_function(name: str):
    """Factory for loss functions."""
    losses = {
        "mse": mse_loss,
        "thresholded": thresholded_mse_loss,
        "binary_cross_entropy": binary_cross_entropy_loss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(losses.keys())}")
    return losses[name]
