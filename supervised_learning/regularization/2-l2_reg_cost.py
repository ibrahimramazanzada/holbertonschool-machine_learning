#!/usr/bin/env python3
'''Regularization'''
import tensorflow as tf


def l2_reg_cost(cost, model):
    '''Calculates the cost of a neural network with L2 regularization'''
    l2_cost = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables])
    l2_cost *= (model.lambtha / 2)
    return cost + l2_cost
