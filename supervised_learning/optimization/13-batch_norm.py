#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''Batch normalization on a matrix Z.'''
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
