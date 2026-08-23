#!/usr/bin/env python3
"""Calculating p affinity for t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """Calculates the P affinity for t-SNE"""

    n, d = X.shape

    # Compute squared pairwise distances
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # Ensure diagonal is exactly 0
    np.fill_diagonal(D, 0)

    # Initialize P affinities matrix to zeros
    P = np.zeros((n, n))

    # Initialize betas to ones
    betas = np.ones((n, 1))

    # Shannon entropy for the given perplexity, base 2
    H = np.log2(perplexity)

    return D, P, betas, H
