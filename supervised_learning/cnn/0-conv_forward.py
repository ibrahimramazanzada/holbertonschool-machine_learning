#!/usr/bin/env python3
"""Convolutional Forward Propagation"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """forward propagation over a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        ph, pw = padding

    new_h = int((h_prev + 2 * ph - kh) / sh) + 1
    new_w = int((w_prev + 2 * pw - kw) / sw) + 1

    A_prev_pad = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant')
    Z = np.zeros((m, new_h, new_w, c_new))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                A_slice = A_prev_pad[:, vert_start:vert_end,
                                     horiz_start:horiz_end, :]
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k], axis=(1, 2, 3))

    Z += b.reshape(1, 1, 1, c_new)
    A = activation(Z)

    return A
