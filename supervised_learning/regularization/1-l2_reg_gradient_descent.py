#!/usr/bin/env python3
'''Regularization'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''Updates the weights and biases of a neural network with L2
    using gradient descent'''
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * W)
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev))
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
