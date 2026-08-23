#!/usr/bin/env python3
"""Calculating p affinity for t-SNE"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low = None
        high = None
        beta_i = betas[i].copy()

        # Distances to all points except i itself
        Di = np.delete(D[i], i)

        Hi, Pi = HP(Di, beta_i)
        Hdiff = Hi - H

        # Binary search for the correct beta
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                low = beta_i.copy()
                if high is None:
                    beta_i = beta_i * 2
                else:
                    beta_i = (beta_i + high) / 2
            else:
                high = beta_i.copy()
                if low is None:
                    beta_i = beta_i / 2
                else:
                    beta_i = (beta_i + low) / 2

            Hi, Pi = HP(Di, beta_i)
            Hdiff = Hi - H

        # Insert Pi back into row i of P, skipping the diagonal
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

        betas[i] = beta_i

    # Symmetrize and normalize
    P = (P + P.T) / (2 * n)

    return P
