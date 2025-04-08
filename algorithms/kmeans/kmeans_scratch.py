"""
@author George Reid-Smith

k-Means Clustering Implementation

"""

import numpy as np


class KMeans:
    def __init__(self, k: int, convergence_threshold: float = 1e-4):
        self.k = k
        self.labels = None
        self.centroids = None
        self.convergence_threshold = convergence_threshold

    def _initialize_centroids(self, X: np.ndarray) -> None:
        """Randomly assign self.k centroids for clusters.

        Args:
            X (np.ndarray): data points to extract centroids

        Raises:
            ValueError: if number of clusters is greater than data points
        """
        if not X.shape[0] > self.k:
            raise ValueError(
                f"Not enough data points ({X.shape[0]}) for {self.k} clusters"
            )

        # Randomize the input
        randomized = np.random.permutation(X.shape[0])

        # Get the centroid indices
        centroid_indices = randomized[: self.k]

        self.centroids = X[centroid_indices]

    def _determine_centroids(self, X: np.ndarray) -> np.ndarray:
        """Returns an array of centroid labels for each point.

        Args:
            X (np.ndarray): data array

        Returns:
            np.ndarray: centroid labels
        """
        distance = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=-1)

        # Indices of centroid closest to datapoint, i.e. centroid label array
        points = np.argmin(distance, axis=1)

        # Update cluster labels
        self.labels = points

        return points

    def _compute_centroid_means(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute the average point for each centroid.

        Args:
            X (np.ndarray): data array
            points (np.ndarray): centroid labels

        Returns:
            np.ndarray: average point for each centroid
        """
        # Centroid mean array
        centroids = np.zeros((self.k, X.shape[1]))

        for i in range(self.k):
            # Average of the points assigned to the centroid
            centroid_mean = X[labels == i].mean(axis=0)
            centroids[i] = centroid_mean

        return centroids

    def _has_converged(self, prev: np.ndarray, new: np.ndarray) -> bool:
        """Determine if clustering has reached convergence - i.e. negligble
        change in centroids.

        Args:
            prev (np.ndarray): previous centroids
            new (np.ndarray): new centroids

        Returns:
            bool: true if converged, false otherwise
        """
        change = np.max(np.linalg.norm((new - prev), axis=1))

        return change < self.convergence_threshold

    def fit(self, X: np.ndarray, max_iterations: int = 10) -> None:
        # Cast as an array
        X = np.asarray(X)

        # Initialize random centroids
        self._initialize_centroids(X)

        for _ in range(max_iterations):
            # Determine closest centroid for each point
            labels = self._determine_centroids(X)

            # Compute average point for each label
            updated_centroids = self._compute_centroid_means(X, labels)

            # End if converged
            if self._has_converged(self.centroids, updated_centroids):
                return

            # Update centroids
            self.centroids = updated_centroids
