#!/usr/bin/env python3
"""Assembling t-SNE"""
import numpy as np


pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.
    """
    n, d = X.shape

    # Reduce dimensionality with PCA first
    X = pca(X, idims)

    # Compute P affinities and apply early exaggeration
    P = P_affinities(X, perplexity=perplexity)
    P = P * 4

    # Initialize Y randomly
    Y = np.random.randn(n, ndims)
    Y_prev = Y.copy()

    for i in range(iterations):
        dY, Q = grads(Y, P)

        if i < 20:
            a = 0.5
        else:
            a = 0.8

        Y_new = Y + lr * dY + a * (Y - Y_prev)
        Y_prev = Y
        Y = Y_new

        # Re-center Y
        Y = Y - np.mean(Y, axis=0)

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        # Stop early exaggeration after 100 iterations
        if i == 100:
            P = P / 4

    return Y
