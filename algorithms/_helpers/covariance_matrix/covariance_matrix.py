import numpy as np
from typing import Literal


def create_covariance_matrix(
    X: np.ndarray, method: Literal["mle", "unbiased"] = "unbiased"
) -> np.ndarray:
    """Creates a covariance matrix from a training matrix.

    Args:
        X (np.ndarray): training matrix
        method: ("mle" or "unbiased") division method

    Raises:
        ValueError: if the training matrix is not 2d
        ValueError: if an incorrect division method is supplied

    Returns:
        np.ndarray: covariance matrix
    """
    if not len(X.shape) == 2:
        raise ValueError("Must be a 2d array")

    # Get total samples
    n_samples = X.shape[0]

    # Center data
    X_centered: np.ndarray = X - np.mean(X, axis=0)

    if method == "mle":
        divisor = n_samples

    elif method == "unbiased":
        divisor = max(1, n_samples - 1)

    else:
        raise ValueError("Method must be 'mle' or 'unbiased'")

    covariance_matrix = (X_centered.T @ X_centered) / divisor

    return covariance_matrix
