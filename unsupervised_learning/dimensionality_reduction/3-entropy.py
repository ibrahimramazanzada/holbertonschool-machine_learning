#!/usr/bin/env python3
"""Calculating entropy"""
import numpy as np


def HP(Di, beta):
    """Calculates the entropy of a distribution for a given precision"""

    # Compute the exponentials of the negative distances scaled by beta
    P = np.exp(-Di * beta)

    # Normalize to get probabilities
    sumP = np.sum(P)
    if sumP == 0:
        H = 0
        P = np.zeros_like(P)
    else:
        P /= sumP
        H = -np.sum(P * np.log2(P + 1e-10))  # Add small value to avoid log(0)

    return H, P
