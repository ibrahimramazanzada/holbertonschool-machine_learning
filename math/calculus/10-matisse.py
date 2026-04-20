#!/usr/bin/env python3
"""This module really does smth"""


def poly_derivative(poly):
    """Calculate derivative of polynomial"""
    return [coef * exp for exp, coef in enumerate(poly)][1:]
