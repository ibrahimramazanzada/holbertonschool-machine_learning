#!/usr/bin/env python3
"""concats every dimension"""


def get_shape(matrix):
    """Returns the shape of a matrix as a list of integers."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis."""
    s1 = get_shape(mat1)
    s2 = get_shape(mat2)

    if len(s1) != len(s2):
        return None

    for i in range(len(s1)):
        if i != axis and s1[i] != s2[i]:
            return None

    if axis == 0:
        return mat1 + mat2
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    new_matrix = []
    for m1_sub, m2_sub in zip(mat1, mat2):
        res = cat_matrices(m1_sub, m2_sub, axis - 1)
        if res is None:
            return None
        new_matrix.append(res)
    return new_matrix
