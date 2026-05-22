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

    def pdf(self, x):
        """calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({d}, 1)".format(d=d))

        d = self.mean.shape[0]
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)

        norm_const = 1.0 / (np.power(2 * np.pi, d / 2) * np.sqrt(cov_det))
        x_diff = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_diff.T, cov_inv), x_diff)

        return float((norm_const * np.exp(exponent))[0][0])
