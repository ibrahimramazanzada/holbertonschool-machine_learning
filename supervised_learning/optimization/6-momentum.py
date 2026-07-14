#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    '''sets up the momentum algorithm in tensorflow, returns optimizer'''
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
