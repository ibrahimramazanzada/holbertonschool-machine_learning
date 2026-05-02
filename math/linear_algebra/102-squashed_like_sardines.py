#!/usr/bin/env python3
"""concats every dimension"""


def cat_matrices(mat1, mat2, axis=0):
    """Concats"""
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if axis == 0:
        return mat1 + mat2
    else:
        if len(mat1) != len(mat2):
            return None

        result = []
        for i in range(len(mat1)):
            concatenated = cat_matrices(mat1[i], mat2[i], axis - 1)
            if concatenated is None:
                return None
            result.append(concatenated)
        return result
    