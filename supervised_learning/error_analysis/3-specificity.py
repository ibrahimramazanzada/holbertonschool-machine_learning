#!/usr/bin/env python3
'''Calculating specificity from confusion matrix.'''
import numpy as np


def specificity(confusion):
    '''Calculate specificity from confusion matrix with lots of classes.'''
    specificity = np.zeros(confusion.shape[0], dtype=float)

    for i in range(confusion.shape[0]):
        true_negatives = (np.sum(confusion) - np.sum(confusion[i, :]) -
                          np.sum(confusion[:, i]) + confusion[i, i])
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]

        if true_negatives + false_positives == 0:
            specificity[i] = 0.0
        else:
            predicted = true_negatives + false_positives
            specificity[i] = true_negatives / predicted

    return specificity
