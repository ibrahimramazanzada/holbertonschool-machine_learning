#!/usr/bin/env python3
"""This module really does smth"""


def poly_derivative(poly):
    """Calculate derivative of polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    if len(poly) == 1:
        return [0]
    return [coef * exp for exp, coef in enumerate(poly)][1:]
