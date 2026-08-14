#!/usr/bin/env python3
"""Convolutional Backward Propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """backward propagation over a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        ph, pw = padding

    new_h = int((h_prev + 2 * ph - kh) / sh) + 1
    new_w = int((w_prev + 2 * pw - kw) / sw) + 1

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)

    db = np.sum(dZ, axis=(0, 1, 2)).reshape(b.shape)

    for i in range(new_h):
        for j in range(new_w):
            vert_start = i * sh
            vert_end = vert_start + kh

            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev_pad[
                :,
                vert_start:vert_end,
                horiz_start:horiz_end,
                :
            ]

            for k in range(c_new):
                dz = dZ[:, i:i + 1, j:j + 1, k:k + 1]

                dA_prev_pad[
                    :,
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ] += W[:, :, :, k] * dz

                dW[:, :, :, k] += np.sum(
                    A_slice * dz,
                    axis=0
                )

    if padding == "same":
        dA_prev = dA_prev_pad[
            :,
            ph:ph + h_prev,
            pw:pw + w_prev,
            :
        ]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
