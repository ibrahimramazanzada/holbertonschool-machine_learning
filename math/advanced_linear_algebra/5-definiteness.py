##!usr/bin/env python3
"""calculates the definiteness of a matrix"""
import numpy as np



def definiteness(matrix):
    """calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.allclose(matrix, matrix.T):
        return None
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
