#!/usr/bin/env python3
"""Perform the Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform Monte Carlo learning and return the updated value estimate."""
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, reward))
            state = new_state

            if terminated or truncated:
                break

        visited = set()
        G = 0

        for state, reward in reversed(episode):
            G = gamma * G + reward

            if state not in visited:
                V[state] = V[state] + alpha * (G - V[state])
                visited.add(state)

    return V
