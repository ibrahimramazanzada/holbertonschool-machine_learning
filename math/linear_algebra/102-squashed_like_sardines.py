#!/usr/bin/env python3
"""concats every dimension"""


def cat_matrices(mat1, mat2, axis=0):
    """"Concatenates"""
    if axis == 0:
        if isinstance(mat1, list) and isinstance(mat2, list):
            if isinstance(mat1[0], list) and isinstance(mat2[0], list):
                if len(mat1[0]) != len(mat2[0]):
                    return None
            return mat1 + mat2
        return None

    if not isinstance(mat1, list) or not isinstance(mat2, list) or len(mat1) != len(mat2):
        return None

    new_matrix = []
    for i in range(len(mat1)):
        res = cat_matrices(mat1[i], mat2[i], axis - 1)
        if res is None:
            return None
        new_matrix.append(res)
    
    return new_matrix
