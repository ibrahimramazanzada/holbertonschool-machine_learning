#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def shuffle_data(X, Y):
    '''shuffles the data points in two matrices the same way'''
    if X.shape[0] != Y.shape[0]:
        return None, None
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
