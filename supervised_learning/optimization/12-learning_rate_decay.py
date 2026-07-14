#!/usr/bin/env python3
'''Optimization'''
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''updates the learning rate using inverse time decay in tensorflow'''
    alpha = tf.train.inverse_time_decay(alpha=alpha, global_step=global_step,
                                        decay_step=decay_step,
                                        decay_rate=decay_rate, staircase=True)
    return alpha
