#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    '''sets up the RMSProp algorithm in tensorflow, returns optimizer'''
    return tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                       rho=beta2, epsilon=epsilon)
