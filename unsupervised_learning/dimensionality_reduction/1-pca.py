#!/usr/bin/env python3
"""PCA implementation"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # SVD on the centered data
    U, S, Vt = np.linalg.svd(X_centered)

    # Take the first ndim principal directions
    W = Vt[:ndim].T

    # Project the centered data onto the new space
    T = X_centered @ W

    return T
