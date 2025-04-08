import numpy as np
from typing import Literal

from _helpers.general import create_covariance_matrix as cov


class GMM:
    def __init__(self, model_type: Literal["supervised"] = "supervised"):
        self.model_type = model_type

    @staticmethod
    def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, covariance: np.ndarray):
        d = len(mean)  # dimensionality
        cov_det = np.linalg.det(covariance)
        cov_inv = np.linalg.inv(covariance)

        mahalanobis_distance = -0.5 * (X - mean).T @ cov_inv @ (X - mean)

        return (
            -0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(cov_det) * mahalanobis_distance
        )

    def _supervised_fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        labels, counts = np.unique(y, return_counts=True)
        params = np.zeros_like((len(labels), 4), dtype=object)

        for i, label in enumerate(labels):
            # Training inputs with label
            X_label = X[y == label]

            # Compute GMM parameters
            probability = counts[i] / n_samples
            mean = np.mean(X_label, axis=0)
            covariance = cov(X=X_label, method="mle")

            # Store in paramters array
            params[i] = (label, probability, mean, covariance)

        self.params = params
        return params

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            return ValueError("X and y must have the same number of samples")

        if self.model_type == "supervised":
            return self._supervised_fit(X, y)

    def predict(self, X_test: np.ndarray):

        log_probs = np.zeros(len(self.params))

        for i, params in enumerate(self.params):
            label, probability, mean, covariance = params

            log_prior = np.log(probability)
            log_likelihood = self._log_gaussian_pdf(X_test, mean, covariance)

            log_probs = log_prior + log_likelihood

        return self.params[np.argmax(log_probs)]

