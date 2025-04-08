import numpy as np


def mean_squared_error_criterion(y_pred: np.ndarray, y_ground: np.ndarray) -> float:

    if y_pred.shape[0] != y_ground.shape[0]:
        raise ValueError(
            "Prediction and ground need to have the same number of samples."
        )

    return np.mean((y_ground - y_pred) ** 2)
