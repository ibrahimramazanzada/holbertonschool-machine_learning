#!/usr/bin/env python3
"""This does smth fr"""


def matrix_transpose(matrix):
    """Transposes a matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
