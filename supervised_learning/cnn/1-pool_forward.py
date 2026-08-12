#!/usr/bin/env python3
"""Pooling Forward Propagation"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """forward propagation over a pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = int((h_prev - kh) / sh) + 1
    new_w = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, new_h, new_w, c_prev))

    for i in range(new_h):
        for j in range(new_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev[:, vert_start:vert_end,
                             horiz_start:horiz_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A
