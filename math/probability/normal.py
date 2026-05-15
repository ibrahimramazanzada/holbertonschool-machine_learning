#!/usr/bin/env python3
"""Normal distribution."""


class Normal:
    """Normal distribution."""

    def __init__(self, data=None, mean=0, stddev=1):
        """Initialize"""

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float((sum((x - self.mean) ** 2 for x in data) /
                                 len(data)) ** 0.5)

    def z_score(self, x):
        """Z-score of x"""

        return (x - self.mean) / self.stddev if self.stddev else 0

    def x_value(self, z):
        """X-value of z"""

        return self.mean + z * self.stddev

    def pdf(self, x):
        """PDF of x"""

        e = 2.7182818285
        pi = 3.1415926536
        return ((1 / (self.stddev * ((2 * pi) ** 0.5))) *
                (e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)))

    def cdf(self, x):
        """
        Approximate CDF using numerical integration of PDF
        """
        # start far in the left tail (approx -infinity)
        start = self.mean - 10 * self.stddev
        step = 0.001

        total = 0.0
        t = start

        while t < x:
            total += self.pdf(t) * step
            t += step

        return total
