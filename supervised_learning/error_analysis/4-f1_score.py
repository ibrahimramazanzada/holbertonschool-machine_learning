#!/usr/bin/env python3
'''Calculating F1 score using precision and sensitivity.'''
import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    p = precision(confusion)
    s = sensitivity(confusion)
    f1 = 2 * (p * s) / (p + s)
    f1[np.isnan(f1)] = 0.0
    return f1
