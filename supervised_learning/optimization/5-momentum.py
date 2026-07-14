#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''updates a variable using the momentum algorithm'''
    if not isinstance(alpha, float) or alpha < 0:
        return None
    if not isinstance(beta1, float) or beta1 < 0 or beta1 >= 1:
        return None
    if not isinstance(var, np.ndarray) or len(var.shape) != 1:
        return None
    if not isinstance(grad, np.ndarray) or grad.shape != var.shape:
        return None
    if not isinstance(v, np.ndarray) or v.shape != var.shape:
        return None
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
