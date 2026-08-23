#!/usr/bin/env python3
"""Calculating cost for t-SNE"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation.
    """
    # Avoid division by 0 / log(0) errors
    Q = np.where(Q < 1e-12, 1e-12, Q)
    P = np.where(P < 1e-12, 1e-12, P)

    # Kullback-Leibler divergence
    C = np.sum(P * np.log(P / Q))

    return C
