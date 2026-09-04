#!/usr/bin/env python3
"""Module that computes a policy using a weight matrix."""
import numpy as np


def policy(matrix, weight):
    """
    Compute the policy with a weight of a matrix.
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=-1, keepdims=True)
