#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize"""

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - (variance / mean)
            self.n = int(round(mean / self.p))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """PMF of k"""

        if k < 0 or k > self.n:
            return 0
        k = int(k)
        p_k = self.p ** k
        q_n_k = (1 - self.p) ** (self.n - k)
        n_k_factorial = 1
        for i in range(1, self.n - k + 1):
            n_k_factorial *= i
        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i
        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i
        return (n_factorial / (k_factorial * n_k_factorial)) * p_k * q_n_k
