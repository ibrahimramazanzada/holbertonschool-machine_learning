#!/usr/bin/env python3
"""Kmeans Clustering initialization of centroids"""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    m, n = X.shape
    centroids = np.zeros((k, n))
    random_indices = np.random.choice(m, size=k, replace=False)
    centroids = X[random_indices]

    return centroids
