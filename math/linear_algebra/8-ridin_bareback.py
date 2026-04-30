#!/usr/bin/env python3
"""what is with the names"""


def mat_mul(mat1, mat2):
    """multiplies 2 matrices"""
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(row, col))
             for col in zip(*mat2)] for row in mat1]
