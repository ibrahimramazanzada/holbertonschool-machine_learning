#!/usr/bin/env python3
"""finding covariance"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain at least 2 data points")

    mean = np.mean(X, axis=0)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
