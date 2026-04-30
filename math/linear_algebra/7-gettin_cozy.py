#!/usr/bin/env python3
"""this do smth cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """"concats matrices"""
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
