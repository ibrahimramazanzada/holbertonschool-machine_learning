#!/usr/bin/env python3
'''Creating confusion matrix from labels and logits.'''
import numpy as np


def create_confusion_matrix(labels, logits):
    '''Create a confusion matrix from labels and logits.'''
    predicted = np.argmax(logits, axis=1)
    true = np.argmax(labels, axis=1)
    num_classes = logits.shape[1]

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(true)):
        true_class = true[i]
        predicted_class = predicted[i]
        confusion_matrix[true_class, predicted_class] += 1

    return confusion_matrix
