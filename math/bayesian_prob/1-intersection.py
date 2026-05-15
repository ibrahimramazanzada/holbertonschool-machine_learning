#!/usr/bin/env python3
"""intersection added"""
import numpy as np


def intersection(x, n, P, Pr):
    """calculates the intersection"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("All values in Pr must sum to 1")

    return likelihood(x, n, P) * Pr


def likelihood(x, n, P):
    """calculates the likelihood"""

    def prob(n, x, P):
        """calculates the probability of x successes in n trials with
        probability P"""
        p_x = P ** x
        q_n_x = (1 - P) ** (n - x)
        n_x_factorial = 1
        for i in range(1, n - x + 1):
            n_x_factorial *= i
        x_factorial = 1
        for i in range(1, x + 1):
            x_factorial *= i
        n_factorial = 1
        for i in range(1, n + 1):
            n_factorial *= i
        return (n_factorial / (x_factorial * n_x_factorial)) * p_x * q_n_x
    arr = []
    for p in P:
        arr.append(prob(n, x, p))
    return np.array(arr)
