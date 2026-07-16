#!/usr/bin/env python3
'''Convolution'''
import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''performs a valid convolution on grayscale images'''
    m, h, w = images.shape
    kh, kw = kernel.shape
    new_h = h - kh + 1
    new_w = w - kw + 1
    conv = np.zeros((m, new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            conv[:, i, j] = np.sum(images[:, i:i + kh, j:j + kw] * kernel,
                                   axis=(1, 2))
    return conv
