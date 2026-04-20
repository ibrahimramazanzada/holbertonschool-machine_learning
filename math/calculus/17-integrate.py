#!/usr/bin/env python3
"""This module really does smth"""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial."""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, (int, float)):
        return None

    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    integral = [C]

    for i, coeff in enumerate(poly):
        value = coeff / (i + 1)
        if value == int(value):
            value = int(value)
        integral.append(value)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
