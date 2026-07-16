#!/usr/bin/env python3
'''Regularization'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''Calculates the cost of a neural network with L2 regularization'''
    l2_cost = 0
    for i in range(L):
        l2_cost += np.sum(np.square(weights[i]))
    l2_cost *= (lambtha / (2 * m))
    return cost + l2_cost
  
