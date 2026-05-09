#!/usr/bin/env python3
"""calculates the determinant"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if matrix == [[]]:
        return 1
    if (not isinstance(matrix, list) or len(matrix) == 0 or
       not isinstance(matrix[0], list)):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    return det
