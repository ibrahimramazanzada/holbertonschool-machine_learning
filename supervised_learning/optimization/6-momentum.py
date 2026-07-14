#!/usr/bin/env python3
'''Optimization'''
import TensorFlow as tf


def create_momentum_op(alpha, beta1):
    '''sets up the momentum algorithm in tensorflow, returns optimizer'''
    return tf.train.MomentumOptimizer(alpha, beta1)
