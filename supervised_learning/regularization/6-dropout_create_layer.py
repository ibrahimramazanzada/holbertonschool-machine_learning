#!/usr/bin/env python3
"""Forward propagation with dropout"""

import numpy as np


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """Creates a layer of a neural network using dropout"""
    if activation not in ['sigmoid', 'tanh', 'relu']:
        return None

    W = np.random.randn(n, prev) * np.sqrt(2 / prev)
    b = np.zeros((n, 1))

    if training:
        D = np.random.rand(n, 1) < keep_prob
        A = np.matmul(W, prev) + b
        A = A * D / keep_prob
    else:
        A = np.matmul(W, prev) + b

    return A