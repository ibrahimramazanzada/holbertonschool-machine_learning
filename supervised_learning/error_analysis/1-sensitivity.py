#!/usr/bin/env python3
'''Calculating sensitivity from confusion matrix.'''
import numpy as np


def sensitivity(confusion):
    '''Calculate sensitivity from confusion matrix with lots of classes.'''
    sensitivity = np.zeros(confusion.shape[0], dtype=float)

    for i in range(confusion.shape[0]):
        true_positives = confusion[i, i]
        false_negatives = np.sum(confusion[i, :]) - true_positives

        if true_positives + false_negatives == 0:
            sensitivity[i] = 0.0
        else:
            predicted = true_positives + false_negatives
            sensitivity[i] = true_positives / predicted

    return sensitivity
