#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''updates a variable using the Adam optimization algorithm'''
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.square(grad)
    v_corrected = v / (1 - np.power(beta1, t))
    s_corrected = s / (1 - np.power(beta2, t))
    var_update = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var_update, v, s
