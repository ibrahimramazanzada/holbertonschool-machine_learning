#!/usr/bin/env python3
"""PCA implementation"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""

    U, S, Vt = np.linalg.svd(X)

    # W is the first ndim rows of Vt, transposed to shape (d, ndim)
    W = Vt[:ndim].T

    return W
