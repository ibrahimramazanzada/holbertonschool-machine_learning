#!/usr/bin/env python3
"""Poisson distribution."""


class Poisson:
    """Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize"""

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """PMF of k"""

        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        lambtha_k = self.lambtha ** k
        e_lambtha = e ** (-self.lambtha)
        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i
        return (lambtha_k * e_lambtha) / k_factorial

    def cdf(self, k):
        """CDF of k"""

        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        e_lambtha = e ** (-self.lambtha)
        cdf = 0
        for i in range(k + 1):
            lambtha_i = self.lambtha ** i
            i_factorial = 1
            for j in range(1, i + 1):
                i_factorial *= j
            cdf += (lambtha_i * e_lambtha) / i_factorial
        return cdf
