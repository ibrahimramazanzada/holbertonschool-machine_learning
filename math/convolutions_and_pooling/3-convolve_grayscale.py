#!/usr/bin/env python3
'''Convolution'''
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = ((h - 1) * sh + kh - h) // 2
        pad_w = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding

    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    new_h = int((h + 2 * pad_h - kh) // sh + 1)
    new_w = int((w + 2 * pad_w - kw) // sw + 1)

    conv = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            conv[:, i, j] = np.sum(padded_images[:, i * sh:i * sh + kh,
                                                 j * sw:j * sw + kw] *
                                   kernel, axis=(1, 2))
    return conv
