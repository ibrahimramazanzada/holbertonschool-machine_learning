#!/usr/bin/env python3
"""This module really does smth"""


def poly_derivative(poly):
    """Calculate derivative of polynomial"""
    if not isinstance(poly, list):
        return None
    if len(poly) < 2:
        return [0]
    return [coef * exp for exp, coef in enumerate(poly)][1:]
