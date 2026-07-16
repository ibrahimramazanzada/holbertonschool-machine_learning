#!/usr/bin/env python3
"""Forward propagation with dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """Creates a layer of a neural network using dropout"""
    
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer)
    dropout_layer = layer(prev, training=training)
    dropout_layer = tf.nn.dropout(dropout_layer, keep_prob)

    return dropout_layer
