#!/usr/bin/env python3
'''Convolution'''
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    '''performs a convolution on images with channels'''
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = np.ceil(((h - 1) * sh + kh - h) / 2).astype(int)
        pw = np.ceil(((w - 1) * sw + kw - w) / 2).astype(int)
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    new_h = int((h + 2 * ph - kh) // sh + 1)
    new_w = int((w + 2 * pw - kw) // sw + 1)

    convolved_images = np.zeros((m, new_h, new_w, nc))

    for i in range(new_h):
        for j in range(new_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            image_slice = padded_images[:, vert_start:vert_end,
                                        horiz_start:horiz_end, :]
            convolved_images[:, i, j] = np.tensordot(image_slice,
                                                     kernels,
                                                     axes=([1, 2, 3],
                                                           [0, 1, 2]))

    return convolved_images
