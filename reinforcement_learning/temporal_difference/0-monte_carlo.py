#!/usr/bin/env python3
"""Performs the Monte Carlo algorithm."""

import numpy as np


def monte_carlo(
        env, V, policy, episodes=5000, max_steps=100,
        alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.
    """
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []

        for step in range(max_steps):
            action = policy(state)
            step_result = env.step(action)
            next_state, reward, terminated, truncated, info = step_result
            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
