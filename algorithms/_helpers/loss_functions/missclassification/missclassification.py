import numpy as np


def misclassification_criterion(labels: np.ndarray) -> float:
    labels = np.asarray(labels)

    if labels.size == 0:
        return 0.0

    counts = np.bincount(labels)
    return 1 - np.max(counts) / labels.size
