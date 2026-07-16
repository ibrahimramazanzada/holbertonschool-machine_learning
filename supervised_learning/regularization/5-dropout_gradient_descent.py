#!/usr/bin/env python3
"""Forward propagation with dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with dropout using gradient descent"""

    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            D = cache["D{}".format(i - 1)]
            dA_prev = np.matmul(W.T, dZ)
            dA_prev = (dA_prev * D) / keep_prob
            dZ = dA_prev * (1 - np.power(A_prev, 2))

        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db
