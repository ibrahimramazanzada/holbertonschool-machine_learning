#!/usr/bin/env python3
'''Convolution'''
import numpy as np


def pool(images, kernel_shape, stride=(1, 1), mode='max'):
    '''performs pooling on images with channels'''
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = int((h - kh) // sh + 1)
    new_w = int((w - kw) // sw + 1)

    pooled_images = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            image_slice = images[:, vert_start:vert_end,
                                 horiz_start:horiz_end, :]

            if mode == 'max':
                pooled_images[:, i, j] = np.max(image_slice,
                                                axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j] = np.mean(image_slice,
                                                 axis=(1, 2))

    return pooled_images
