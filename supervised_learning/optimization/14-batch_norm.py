#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''Creates a batch normalization layer for a nn in tf'''
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)
    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta,
                                       gamma, epsilon)
    if activation is not None:
        return activation(Z_norm)
    return Z_norm
