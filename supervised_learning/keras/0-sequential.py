#!/usr/bin/env python3
'''Sequential model implementation with keras'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Builds a neural network with the Keras library'''
    sequential_model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            sequential_model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            sequential_model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            ))
        if keep_prob < 1:
            sequential_model.add(K.layers.Dropout(1 - keep_prob))
    return sequential_model
