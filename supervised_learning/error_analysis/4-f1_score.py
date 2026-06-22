#!/usr/bin/env python3
'''Calculating F1 score using precision and sensitivity.'''
import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    '''Calculate F1 score from confusion matrix with lots of classes.'''
    p = precision(confusion)
    s = sensitivity(confusion)
    f1 = 2 * (p * s) / (p + s)
    f1[np.isnan(f1)] = 0.0
    return f1
