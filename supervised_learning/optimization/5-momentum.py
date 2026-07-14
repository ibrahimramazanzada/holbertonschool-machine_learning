#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''updates a variable using the momentum algorithm'''
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
