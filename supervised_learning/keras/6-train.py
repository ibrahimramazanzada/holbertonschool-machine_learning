#!/usr/bin/env python3
'''Input model implementation with keras'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    '''Trains a model using mini-batch gradient descent'''
    if validation_data and early_stopping:
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=[early_stopping]
                          if early_stopping and validation_data else None,
                          verbose=verbose, shuffle=shuffle)
    return history
