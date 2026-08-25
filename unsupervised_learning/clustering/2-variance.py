#!/usr/bin/env python3
"""Calculates the intra-cluster variance of a dataset"""
import numpy as np


def variance(X, C):
    """
    Calculates the intra-cluster variance of a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    labels = np.argmin(distances, axis=1)

    variance = 0.0
    for i in range(C.shape[0]):
        points = X[labels == i]
        if points.shape[0] > 0:
            variance += np.sum(np.linalg.norm(points - C[i], axis=1) ** 2)

    return variance
