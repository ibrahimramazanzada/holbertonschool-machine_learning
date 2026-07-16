#!/usr/bin/env python3
'''Regularization'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''Creates the forward propagation graph for the neural network
       using dropout'''
    A = X
    for i in range(1, L):
        Z = np.add(np.dot(A, weights['W' + str(i)]), weights['b' + str(i)])
        A = np.tanh(Z)
        D = np.random.binomial(1, keep_prob, size=A.shape)
        A = A * D / keep_prob
    ZL = np.add(np.dot(A, weights['W' + str(L)]), weights['b' + str(L)])
    AL = np.softmax(ZL)
    return AL
