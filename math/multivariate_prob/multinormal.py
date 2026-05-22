#!/usr/bin/env python3
"""multivariate normal distribution"""
import numpy as np


class MultiNormal:
    """multivariate normal distribution"""

    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.array([np.sum(data.T, axis=0) / n])
        self.cov = (np.matmul((data.T - self.mean).T, (data.T - self.mean)) /
                    (n - 1))
        self.mean = self.mean.T
