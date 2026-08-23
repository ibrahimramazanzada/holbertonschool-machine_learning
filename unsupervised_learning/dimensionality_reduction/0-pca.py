#!/usr/bin/env python3
"""PCA"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""

    U, S, Vt = np.linalg.svd(X)

    # Cumulative variance explained by each singular value
    total_var = np.sum(S)
    cum_var = np.cumsum(S) / total_var

    # Find the smallest number of components that maintain 'var' fraction
    nd = np.argwhere(cum_var >= var)[0, 0] + 1

    # W is the first nd rows of Vt, transposed to shape (d, nd)
    W = Vt[:nd].T

    return W
