#!/usr/bin/env python3
"""Calculating q affinity for t-SNE"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.
    """
    n, ndim = Y.shape

    Q, num = Q_affinities(Y)

    PQ_diff = P - Q
    dY = np.zeros((n, ndim))

    for i in range(n):
        dY[i] = np.sum(
            (PQ_diff[:, i] * num[:, i])[:, np.newaxis] * (Y[i] - Y),
            axis=0
        )

    return dY, Q
