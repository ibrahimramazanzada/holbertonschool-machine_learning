#!/usr/bin/env python3
"""Kmeans Clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids
    low = X.min(axis=0)
    high = X.max(axis=0)
    centroids = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids based on the mean of assigned points
        new_centroids = np.array([X[labels == i].mean(axis=0)
                                  for i in range(k)])

        # Check for empty clusters and reinitialize if necessary
        for i in range(k):
            if np.isnan(new_centroids[i]).any():
                new_centroids[i] = np.random.uniform(low, high, size=(d,))

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels
