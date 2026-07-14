#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Updates the learning rate using inverse time decay in tensorflow."""
    return tf.compat.v1.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=tf.compat.v1.train.get_global_step(),
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
