#!/usr/bin/env python3
"""Calculating p affinity for t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """Calculates the P affinity for t-SNE"""

    (n, d) = X.shape
    D = (np.sum(X**2, axis=1).reshape((n, 1)) +
         np.sum(X**2, axis=1) - 2 * np.dot(X, X.T))
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        Hdiff = H - logU
        tries = 0

        while np.abs(Hdiff) > 1e-5 and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] *= 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] /= 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    P = (P + P.T) / (2 * n)
    return P


def Hbeta(D=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row"""
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P
