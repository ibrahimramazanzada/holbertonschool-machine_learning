#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Updates the learning rate using inverse time decay in tensorflow."""
    global_step = tf.Variable(0, trainable=False)
    alpha = tf.compat.v1.train.inverse_time_decay(alpha, global_step,
                                                  decay_step, decay_rate,
                                                  staircase=True)
    return alpha
