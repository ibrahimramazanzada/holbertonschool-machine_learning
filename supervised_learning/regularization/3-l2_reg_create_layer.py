#!/usr/bin/env python3
'''Regularization'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''Creates a tensorflow layer that includes L2 regularization'''
    kernel_regularizer = tf.keras.regularizers.l2(lambtha)
    kernel_initializer = (tf.keras.initializers.VarianceScaling
                          (scale=2.0, mode=("fan_avg")))
    return tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(prev)
