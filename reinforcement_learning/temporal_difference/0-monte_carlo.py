#!/usr/bin/env python3
"""Perform the Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform Monte Carlo learning and return the updated value estimate."""
    for _ in range(episodes):
        state, _ = env.reset()
        states = []
        rewards = []

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            rewards.append(reward)

            state = new_state

            if terminated or truncated:
                break

        for i in range(len(states)):
            G = 0

            for j in range(i, len(rewards)):
                G += gamma ** (j - i) * rewards[j]

            state = states[i]

            V[state] = V[state] + alpha * (G - V[state])

    return V
