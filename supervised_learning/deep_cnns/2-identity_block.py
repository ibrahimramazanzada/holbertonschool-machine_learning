#!/usr/bin/env python3
"""Image recognition 2015 competition - Identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block for a ResNet architecture"""

    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), padding='same',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
