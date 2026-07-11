#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def normalization_constants(X):
    '''calculates the normalization constants of a matrix'''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
