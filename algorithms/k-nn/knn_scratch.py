"""
@author George Reid-Smith

k-NN Classification

"""

import numpy as np
from typing import Literal


class KNN:
    def __init__(
        self,
        k: int = 3,
        mode: Literal["classification", "regression"] = "classification",
    ):
        self.k = k
        self.mode = mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Set the x and y training variables.

        Args:
            X (np.ndarray): training data input
            y (np.ndarray): training data output

        Raises:
            ValueError: if X and y have different number of samples
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if not X.shape[0] == y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.X_train = X
        self.y_train = y

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Ensure array
        X_test = np.asarray(X_test)

        # Compute the distance between
        distances = np.linalg.norm(self.X_train[:, np.newaxis] - X_test, axis=2)
        print(distances)

        k_indices = np.argsort(distances, axis=1)[: self.k, :]
        print(k_indices)

        if self.mode == "classification":
            return np.array(
                [
                    np.argmax(np.bincount(self.y_train[k_indices[:, i]]))
                    for i in range(X_test.shape[0])
                ]
            )

        elif self.mode == "regression":
            return np.array(
                [
                    np.argmax(np.mean(self.y_train[k_indices[:, i]]))
                    for i in range(X_test.shape[0])
                ]
            )

        else:
            raise ValueError("Invalid mode. Choose 'classification' or 'regression'")
