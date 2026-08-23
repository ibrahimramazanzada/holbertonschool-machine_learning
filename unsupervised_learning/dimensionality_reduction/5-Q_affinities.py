#!/usr/bin/env python3
"""Calculating q affinity for t-SNE"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities.
    """

    # Compute squared pairwise distances in low-dimensional space
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    # Numerator: Student's t-distribution kernel (1 dof)
    num = 1 / (1 + D)

    # Diagonal should not contribute
    np.fill_diagonal(num, 0)

    # Normalize
    Q = num / np.sum(num)

    return Q, num
