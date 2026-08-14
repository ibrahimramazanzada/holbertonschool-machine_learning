#!/usr/bin/env python3
"""lenet5 architecture implementation"""
from tensorflow import keras as K


def lenet5(X):
    """builds a modified version of the LeNet-5 architecture using keras"""

    init = K.initializers.he_normal(seed=0)

    model = K.Sequential()

    model.add(K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init,
        input_shape=X.shape[1:]
    ))

    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    ))

    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    ))

    model.add(K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    ))

    model.add(K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    ))

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
