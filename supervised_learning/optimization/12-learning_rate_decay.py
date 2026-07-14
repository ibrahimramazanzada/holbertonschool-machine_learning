#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Updates the learning rate using inverse time decay in tensorflow."""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        alpha=alpha,
        decay_step=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
