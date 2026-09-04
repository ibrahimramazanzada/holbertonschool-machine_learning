##!/usr/bin/env python3
"""Perform Q-learning on a FrozenLake environment."""

import numpy as np


epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning and return the updated Q-table and rewards."""
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = (
                (1 - alpha) * Q[state, action]
                + alpha * (reward + gamma * np.max(Q[new_state]))
            )

            state = new_state
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        epsilon = max(
            min_epsilon,
            epsilon * (1 - epsilon_decay)
        )

    return Q, total_rewards

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(Q)
split_rewards = np.split(np.array(total_rewards), 10)
for i, rewards in enumerate(split_rewards):
    print((i+1) * 500, ':', np.mean(rewards))