#!/usr/bin/env python3
"""Continuous posterior probability over a range"""

from scipy import special


def posterior(x, n, p1, p2):
    """computes posterior probability over range [p1, p2]"""

    # validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p1 >= p2:
        raise ValueError("p2 must be greater than p1")

    # Beta posterior parameters
    alpha = x + 1
    beta = n - x + 1

    # CDF difference using incomplete beta function
    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)
    return cdf_p2 - cdf_p1
