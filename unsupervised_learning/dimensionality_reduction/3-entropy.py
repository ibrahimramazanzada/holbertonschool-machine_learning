#!/usr/bin/env python3
"""Calculating entropy"""
import numpy as np


def HP(Di, beta):
    """Calculates the entropy of a distribution for a given precision"""

    # Compute the exponentials of the negative distances scaled by beta
    Pi = np.exp(-Di * beta)

    # Normalize to get the P affinities
    sum_Pi = np.sum(Pi)
    Pi = Pi / sum_Pi

    # Shannon entropy, base 2
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
