#!/usr/bin/env python3
"""This module really does smth"""


def summation_i_squared(n):
    """Calculate the first n**2"""
    if n is not int or n < 1:
        return None
    if n == 1:
        return 1
    return n ** 2 + summation_i_squared(n - 1)
