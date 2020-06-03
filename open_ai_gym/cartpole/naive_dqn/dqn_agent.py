"""
Naive DQN implementaion
"""

import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from utils.utils import plot_learning_curve


class LinearDeepNetwork(nn.Module):
    """
    LinearDeepNetwork Class
    """
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):  # pylint: disable=arguments-differ
        """
        Forward Propogation Algo
        """

        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions


class QAgent:
    """
    Q_agent class
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, input_dims, n_actions, **kwargs):
        # hyper params
        self.lr = kwargs.get('lr', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_min = kwargs.get('eps_min', 0.01)
        self.eps_dec = kwargs.get('eps_dec', 1e-5)
        self.epsilon = kwargs.get('eps_start', 1.0)

        self.n_actions = n_actions
        self.input_dims = input_dims

        self.action_space = list(range(n_actions))

        self.Q = LinearDeepNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        """
        choose_action based on epsilon greedy method
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)

            # item() converts pytorch tensor to numpy array
            action = T.argmax(actions).item()

        return action

    def decrement_epsilon(self):
        """
        decrement_epsilon
        """
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        """
        learn based on the bellman equation
        """
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        action = T.tensor(action).to(self.Q.device)
        reward = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[action]

        q_next = self.Q.forward(states_).max()

        q_target = reward + self.gamma*q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()


def run_dqn_training():
    """
    Trainig environemt
    """
    # pylint: disable=too-many-locals
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = QAgent(input_dims=env.observation_space.shape,
                   n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_

        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode: {i} |"
                  f"Score: {score: .1f} |"
                  f"Avg Score: {avg_score: .1f} |"
                  f"Epsilon: {agent.epsilon: .2f}")
    filename = 'plots/cartpole_naive_dqn.png'

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
