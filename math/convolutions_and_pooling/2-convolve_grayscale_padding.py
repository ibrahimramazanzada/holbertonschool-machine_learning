#!/usr/bin/env python3
'''Convolution'''
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''performs a convolution on grayscale images with custom padding'''
    m, h, w = images.shape
    kh, kw = kernel.shape
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')
    conv = np.zeros((m, h + 2 * pad_h - kh + 1, w + 2 * pad_w - kw + 1))
    for i in range(conv.shape[1]):
        for j in range(conv.shape[2]):
            conv[:, i, j] = np.sum(padded_images[:, i:i + kh, j:j + kw] *
                                   kernel, axis=(1, 2))
    return conv
