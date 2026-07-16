#!/usr/bin/env python3
"""Forward propagation with dropout"""

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early"""
    if cost == opt_cost:
        count = 0
    elif cost < opt_cost + threshold:
        count += 1
    elif cost > opt_cost + threshold:
        count += 1
    else:
        count = 0

    if count >= patience:
        return (True, count)

    return (False, count)
