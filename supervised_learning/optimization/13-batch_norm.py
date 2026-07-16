#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''Batch normalization on a matrix Z.

    Z: numpy.ndarray of shape (m, n)
    gamma: numpy.ndarray of shape (1, n) or (n,)
    beta: numpy.ndarray of shape (1, n) or (n,)
    epsilon: small float added to variance for numerical stability

    Returns: normalized Z of same shape as Z
    '''
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
