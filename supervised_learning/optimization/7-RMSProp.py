#!/usr/bin/env python3
'''Optimization'''
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''updates a variable using the RMSProp optimization algorithm'''
    s = beta2 * s + (1 - beta2) * np.square(grad)
    var_update = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var_update, s
