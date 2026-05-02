#!/usr/bin/env python3
"""concats every dimension"""


def cat_matrices(mat1, mat2, axis=0):
    """Concats"""
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if axis == 0:
            if isinstance(mat1, list) and isinstance(mat2, list):
                if isinstance(mat1[0], list) and isinstance(mat2[0], list):
                    if len(mat1[0]) != len(mat2[0]):
                        return None
                return mat1 + mat2
            return None
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
    