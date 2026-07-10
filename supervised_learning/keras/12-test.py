#!/usr/bin/env python3
'''Model implementation with keras'''
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''Tests a model using mini-batch processing'''
    return network.evaluate(x=data, y=labels, verbose=verbose)
