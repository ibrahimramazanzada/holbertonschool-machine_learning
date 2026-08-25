#!/usr/bin/env python3
"""Image recognition 2015 competition - Identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block for a ResNet architecture."""
    F1, F2, F3 = filters

    X = K.layers.Conv2D(F1, (1, 1), padding='same',
                        kernel_initializer='he_normal')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F2, (3, 3), padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (1, 1), padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X_shortcut = A_prev
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)
    return X
