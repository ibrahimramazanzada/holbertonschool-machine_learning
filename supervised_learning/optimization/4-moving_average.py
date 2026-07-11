#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def moving_average(data, beta):
    '''calculates the weighted moving average of a data set with bias correction'''
    if not isinstance(data, np.ndarray) or len(data.shape) != 1:
        return None
    if not isinstance(beta, float) or beta < 0 or beta >= 1:
        return None
    moving_averages = np.zeros(data.shape)
    moving_averages[0] = data[0]
    for i in range(1, data.shape[0]):
        moving_averages[i] = beta * moving_averages[i - 1] + (1 - beta) * data[i]
        moving_averages[i] /= (1 - beta ** (i + 1))
    return moving_averages
