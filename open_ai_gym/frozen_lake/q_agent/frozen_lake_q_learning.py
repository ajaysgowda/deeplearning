"""
FROZEN LAKE Q LEARNING
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

from .q_agent import QAgent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = QAgent(n_actions=4, n_states=16, alpha=0.001, gamma=0.9,
                   eps_start=1.0, eps_min=0.01, eps_dec=0.9999995)
    scores = []
    win_pct_list = []
    N_GAMES = 500000

    for i in range(N_GAMES):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score += reward
            observation = observation_
        scores.append(score)
        if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
        if i % 1000 == 0:
            print(f"episode {i}; win_pct {win_pct: .2f} \
                    epsilon {agent.epsilon: .2f}")

    plt.plot(win_pct_list)
    plt.show()
