#!/usr/bin/env python3
'''Sequential model implementation with keras'''
import numpy as np
import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Builds a neural network with the Keras library'''
    sequential_model = keras.Sequential()
    for i in range(len(layers)):
        if i == 0:
            sequential_model.add(keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            sequential_model.add(keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha)
            ))
        if keep_prob < 1:
            sequential_model.add(keras.layers.Dropout(1 - keep_prob))
    return sequential_model
