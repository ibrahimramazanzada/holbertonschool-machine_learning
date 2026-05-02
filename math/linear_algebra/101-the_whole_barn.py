#!/usr/bin/env python3
"""the whole barn"""


def add_matrices(mat1, mat2):
    """Adds matrices"""
    if type(mat1) is not list or type(mat2) is not list:
        return None

    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        item1 = mat1[i]
        item2 = mat2[i]

        if type(item1) is list and type(item2) is list:
            summed = add_matrices(item1, item2)
            if summed is None:
                return None
            result.append(summed)
        elif type(item1) is not list and type(item2) is not list:
            result.append(item1 + item2)
        else:
            return None

    return result
