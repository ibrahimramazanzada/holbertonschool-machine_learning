#!/usr/bin/env python3
'''Input model implementation with keras'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    '''Trains a model using mini-batch gradient descent'''
    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          early_stopping=early_stopping, patience=patience,
                          verbose=verbose, shuffle=shuffle)
    return history
