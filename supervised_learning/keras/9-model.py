#!/usr/bin/env python3
'''Model implementation with keras'''
import tensorflow.keras as K


def save_model(network, filename):
    '''Saves an entire model'''
    network.save(filename)

def load_model(filename):
    '''Loads an entire model'''
    return K.models.load_model(filename)
