#!/usr/bin/env python3
'''Calculating precision from confusion matrix.'''
import numpy as np


def precision(confusion):
    '''Calculate precision from confusion matrix with lots of classes.'''
    precision = np.zeros(confusion.shape[0], dtype=float)

    for i in range(confusion.shape[0]):
        true_positives = confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - true_positives

        if true_positives + false_positives == 0:
            precision[i] = 0.0
        else:
            predicted = true_positives + false_positives
            precision[i] = true_positives / predicted

    return precision
