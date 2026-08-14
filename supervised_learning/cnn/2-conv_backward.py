#!/usr/bin/env python3
"""Convolutional Backward Propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """backward propagation over a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    _, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = max((h_new - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_new - 1) * sw + kw - w_prev, 0)

        ph_before = pad_h // 2
        ph_after = pad_h - ph_before

        pw_before = pad_w // 2
        pw_after = pad_w - pw_before

    elif padding == "valid":
        ph_before = ph_after = 0
        pw_before = pw_after = 0

    else:
        raise ValueError("padding must be 'same' or 'valid'")

    A_prev_pad = np.pad(
        A_prev,
        (
            (0, 0),
            (ph_before, ph_after),
            (pw_before, pw_after),
            (0, 0)
        ),
        mode="constant"
    )

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)

    db = np.sum(dZ, axis=(0, 1, 2)).reshape(b.shape)

    for i in range(h_new):
        for j in range(w_new):
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
                dz = dZ[:, i:i+1, j:j+1, k:k+1]

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
            ph_before:ph_before + h_prev,
            pw_before:pw_before + w_prev,
            :
        ]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
