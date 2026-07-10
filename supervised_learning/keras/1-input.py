#!/usr/bin/env python3
'''Input model implementation with keras'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Builds a neural network with the Keras library'''
    input_layer = K.Input(shape=(nx,))
    x = input_layer
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        if keep_prob < 1 and i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs=input_layer, outputs=x)
    return model
