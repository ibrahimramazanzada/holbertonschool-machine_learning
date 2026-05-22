#!/usr/bin/env python3
"""calculating correlation matrix"""
import numpy as np


def correlation(C):
    """calculates the correlation matrix from a covariance matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    d, d2 = C.shape

    if d != d2 or len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")

    diag = np.diag(C)

    stddev = np.sqrt(diag)
    corr = C / np.outer(stddev, stddev)

    return corr
