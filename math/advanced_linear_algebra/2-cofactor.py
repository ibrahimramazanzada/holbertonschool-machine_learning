#!/usr/bin/env python3
"""calculates the cofactor of a matrix"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    return det


def cofactor(matrix):
    """calculates the cofactor of a matrix"""
    if (not isinstance(matrix, list) or len(matrix) == 0):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    for row in matrix:
        if (not isinstance(row, list)):
            raise TypeError("matrix must be a list of lists")
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j + 1:]
                          for row in matrix[:i] + matrix[i + 1:]]
            cofactor_row.append(((-1) ** (i + j)) * determinant(sub_matrix))
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix
