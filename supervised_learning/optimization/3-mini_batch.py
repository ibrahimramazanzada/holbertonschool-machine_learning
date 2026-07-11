#!/usr/bin/env python3
'''Optimization'''
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    '''creates mini-batches from two matrices'''
    if X.shape[0] != Y.shape[0]:
        return None, None
    if batch_size <= 0 or batch_size > X.shape[0]:
        return None, None
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    mini_batches_X = []
    mini_batches_Y = []
    mini_batches = []
    for i in range(0, X.shape[0], batch_size):
        mini_batches_X = X_shuffled[i:i + batch_size]
        mini_batches_Y = Y_shuffled[i:i + batch_size]
        mini_batches.append((mini_batches_X, mini_batches_Y))
    return mini_batches
