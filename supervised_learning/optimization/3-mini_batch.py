#!/usr/bin/env python3
'''Optimization'''
shuffle_data = __import__('2-shuffle_data').shuffle_data


def mini_batch(X, Y, batch_size):
    '''creates mini-batches from two matrices'''
    if X.shape[0] != Y.shape[0]:
        return None, None
    if batch_size <= 0 or batch_size > X.shape[0]:
        return None, None
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    mini_batches_X = []
    mini_batches_Y = []
    for i in range(0, X.shape[0], batch_size):
        if i + batch_size > X.shape[0]:
            mini_batches_X.append(X_shuffled[i:X.shape[0]])
            mini_batches_Y.append(Y_shuffled[i:Y.shape[0]])
        else:
            mini_batches_X.append(X_shuffled[i:i + batch_size])
            mini_batches_Y.append(Y_shuffled[i:i + batch_size])
    return mini_batches_X, mini_batches_Y
