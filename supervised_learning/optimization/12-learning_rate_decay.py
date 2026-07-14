#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''updates the learning rate using inverse time decay in tensorflow'''
    # Global step variable (counts training steps)
    global_step = tf.Variable(0, trainable=False)

    # Inverse time decay (stepwise)
    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True  # ensures stepwise decay
    )

    return learning_rate(global_step)
