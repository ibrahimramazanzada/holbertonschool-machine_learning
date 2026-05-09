#!/usr/bin/env python3
"""calculates the minor of a matrix"""


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


def minor(matrix):
    """calculates the minor of a matrix"""
    if (not isinstance(matrix, list) or len(matrix) == 0):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    for row in matrix:
        if (not isinstance(row, list)):
            raise TypeError("matrix must be a list of lists")
    if n != len(matrix[0]) or n == 0 or len(matrix[0]) == 0 or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j + 1:]
                          for row in matrix[:i] + matrix[i + 1:]]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    return minor_matrix
