#!usr/bin/env python3
"""W-watch out"""


def np_slice(matrix, axes={}):
    """slices"""
    slices = [slice(None)] * len(matrix.shape)
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
