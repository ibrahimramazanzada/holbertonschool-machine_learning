#!/usr/bin/env python3
"""Forward propagation with dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """Creates a layer of a neural network using dropout"""
    
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer)
    X = layer(prev)
    dropout_layer = tf.keras.layers.Dropout(1 - keep_prob)

    return dropout_layer(X, training=training)
