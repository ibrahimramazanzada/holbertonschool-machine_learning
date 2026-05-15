#!/usr/bin/env python3
"""binomial likelihood"""
import numpy as np


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
    if n <= 0 or not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    if x < 0 or not isinstance(x, int):
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    arr = []
    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
        arr.append(prob(n, x, p))
    return np.array(arr)
