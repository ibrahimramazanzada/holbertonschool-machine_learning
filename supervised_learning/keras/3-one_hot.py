#!/usr/bin/env python3
'''Input model implementation with keras'''
import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''Converts a numeric label vector into a one-hot matrix'''
    return K.utils.to_categorical(labels, num_classes=classes)
