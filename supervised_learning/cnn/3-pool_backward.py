#!/usr/bin/env python3
"""Pooling Backward Propagation"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """backward propagation over a pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = int((h_prev - kh) / sh) + 1
    new_w = int((w_prev - kw) / sw) + 1

    dA_prev = np.zeros_like(A_prev)

    for i in range(new_h):
        for j in range(new_w):
            vert_start = i * sh
            vert_end = vert_start + kh

            horiz_start = j * sw
            horiz_end = horiz_start + kw

            for k in range(c_prev):
                if mode == 'max':
                    A_slice = A_prev[
                        :,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        k
                    ]
                    mask = (A_slice == np.max(A_slice, axis=(1, 2),
                                              keepdims=True))
                    dA_prev[
                        :,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        k
                    ] += mask * dA[:, i:i + 1, j:j + 1, k]

                elif mode == 'avg':
                    da = dA[:, i:i + 1, j:j + 1, k]
                    average = da / (kh * kw)
                    dA_prev[
                        :,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        k
                    ] += np.ones((m, kh, kw)) * average

    return dA_prev
