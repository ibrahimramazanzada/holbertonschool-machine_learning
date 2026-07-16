#!/usr/bin/env python3
"""Forward propagation with dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Creates the forward propagation graph using dropout"""

    cache = {}
    cache["A0"] = X
    A = X

    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]

        Z = np.matmul(W, A) + b

        if i == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache["D{}".format(i)] = D

        cache["A{}".format(i)] = A

    return cache
