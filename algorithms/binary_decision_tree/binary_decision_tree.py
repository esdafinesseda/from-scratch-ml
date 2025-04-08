import numpy as np
from typing import Literal, Dict, Callable
from _helpers.loss_functions import (
    misclassification_criterion,
    entropy_criterion,
    gini_index_criterion,
    mean_squared_error_criterion,
)


class Node:
    def __init__(self, data_points: np.ndarray, feature_idx: int, feature_value: float):
        self.data_points = data_points
        self.feature_idx = feature_idx
        self.feature_value = feature_value
        self.left = None
        self.right = None


class BinaryDecisionTree:
    def __init__(
        self,
        max_depth: int = 10,
        type: Literal["classification", "regression"] = "regression",
        criterion: Literal["entropy", "gini", "missclassification"] = "entropy",
    ):
        criterion_functions: Dict[str, Callable[[np.array], float]] = {
            "entropy": entropy_criterion,
            "gini": gini_index_criterion,
            "missclassification": misclassification_criterion,
        }

        if criterion not in criterion_functions:
            raise ValueError(f"Incorrect criterion specified: {criterion}")

        self.regression_criterion = mean_squared_error_criterion
        self.criterion = criterion_functions[criterion]
        self.type = type
        self.max_depth = max_depth

    @staticmethod
    def _partition(X: np.ndarray, input_index: int, split_index: int):
        # Value to partition the input
        split_value = X[input_index, split_index]

        n1_indices = np.array([], dtype=int)
        n2_indices = np.array([], dtype=int)

        for i in range(X.size):
            if X[i, split_index] <= split_value:
                n1_indices = np.append(n1_indices, i)

            else:
                n2_indices = np.append(n2_indices, i)

        return n1_indices, n2_indices

    def _classification_loss(
        self, X: np.ndarray, y: np.ndarray, input_index: int, split_index: int
    ) -> float:
        n1_indices, n2_indices = self._partition(X, input_index, split_index)

        n1_labels = y[n1_indices]
        n2_labels = y[n2_indices]

        n1_loss = n1_labels.size * self.criterion(n1_labels)
        n2_loss = n2_labels.size * self.criterion(n2_labels)

        return n1_loss + n2_loss

    def _regression_loss(
        self, X: np.ndarray, y: np.ndarray, input_index: int, split_index: int
    ) -> float:
        n1_indices, n2_indices = self._partition(X, input_index, split_index)

        n1_actual = y[n1_indices]
        n2_actual = y[n2_indices]

        n1_pred = np.full_like(n1_actual, fill_value=np.mean(y[n1_indices]))
        n2_pred = np.full_like(n2_actual, fill_value=np.mean(y[n2_indices]))

        n1_loss = n1_actual.size * self.regression_criterion(n1_pred, n1_actual)
        n2_loss = n2_actual.size * self.regression_criterion(n2_pred, n2_actual)

        return n1_loss + n2_loss

    def _split(self, X: np.ndarray, y: np.ndarray):
        split_losses = np.zeros_like(X)

        for i in X.shape[0]:
            for j in X.shape[1]:
                if self.type == "regression":
                    split_losses[i, j] = self._regression_loss(
                        X, y, input_index=i, split_index=j
                    )

                elif self.type == "classifcation":
                    split_losses[i, j] = self._classification_loss(
                        X, y, input_index=i, split_index=j
                    )

        min_loss_indices = np.unravel_index(np.argmin(split_losses), split_losses.shape)
        min_loss = split_losses[min_loss_indices]

        if min_loss < self.threshold:
            return

    def train(self, X: np.ndarray, y: np.ndarray):
        for input, output in zip(X, y):
            for feature in input:

                loss = self.criterion(y)
