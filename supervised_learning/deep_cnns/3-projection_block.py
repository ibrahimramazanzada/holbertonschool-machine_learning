#!/usr/bin/env python3
"""Image recognition 2015 competition - Projection block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Build a projection block for a ResNet architecture"""

    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='same',
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

    A_shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                                 padding='same',
                                 kernel_initializer=init)(A_prev)
    A_shortcut = K.layers.BatchNormalization(axis=3)(A_shortcut)

    X = K.layers.Add()([X, A_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
