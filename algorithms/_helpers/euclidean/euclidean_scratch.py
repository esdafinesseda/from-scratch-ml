"""
@author George Reid-Smith

Euclidean Distance Implementation

"""

import numpy as np


def euclidean_distance(p1: np.array, p2: np.array) -> float:
    """Computes Euclidean Distance between two points.

    Args:
        p1 (np.array): first point
        p2 (np.array): second point

    Raises:
        ValueError: if the points do not have the same shape

    Returns:
        float: Euclidean distance between p1 and p2
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    if p1.shape != p2.shape:
        raise ValueError("Input arrays must have the same shape")

    return np.sqrt(np.sum((p1 - p2) ** 2))
