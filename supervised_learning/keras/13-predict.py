#!/usr/bin/env python3
'''Model implementation with keras'''
import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''Makes predictions using a neural network'''
    return network.predict(x=data, verbose=verbose)
