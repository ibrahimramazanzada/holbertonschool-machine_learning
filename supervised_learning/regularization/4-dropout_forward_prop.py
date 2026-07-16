#!/usr/bin/env python3
'''Regularization'''
import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    '''Creates the forward propagation graph for the neural network
       using dropout'''
    A = X
    for i in range(1, L):
        Z = tf.add(tf.matmul(A, weights['W' + str(i)]), weights['b' + str(i)])
        A = tf.nn.tanh(Z)
        D = tf.nn.dropout(A, keep_prob)
        A = tf.multiply(D, 1 / keep_prob)
    ZL = tf.add(tf.matmul(A, weights['W' + str(L)]), weights['b' + str(L)])
    AL = tf.nn.softmax(ZL)
    return AL
