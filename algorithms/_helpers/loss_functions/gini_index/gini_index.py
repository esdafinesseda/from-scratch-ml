import numpy as np


def gini_index_criterion(labels: np.ndarray):
    if len(labels) == 0:
        return 0.0

    labels = np.asarray(labels)
    n_samples = len(labels)

    _, label_counts = np.unique(labels, return_counts=True)

    gini_index = 0.0

    for count in label_counts:
        p = count / n_samples

        gini_index += p * (1 - p)

    return gini_index
