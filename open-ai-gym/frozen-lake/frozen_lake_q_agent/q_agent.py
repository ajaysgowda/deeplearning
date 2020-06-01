"""
QAgent
"""
import numpy as np


class QAgent:
    """
    Q_agent class
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, n_actions, n_states, **kwargs):
        # hyper params
        self.alpha = kwargs.get('alpha', 0.001)
        self.gamma = kwargs.get('gamma', 0.999)
        self.eps_min = kwargs.get('eps_min', 0.01)
        self.eps_dec = kwargs.get('eps_dec', 0.999)
        self.epsilon = kwargs.get('eps_start', 1.0)

        self.n_actions = n_actions
        self.n_states = n_states

        self.q_table = {}
        self.init_q_table()

    def init_q_table(self):
        """
        init q table based on action and states
        """
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.q_table[(state, action)] = 0.0

    def choose_action(self, state):
        """
        choose_action based on epsilon greedy method
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(list(range(self.n_actions)))
        else:
            actions = np.array([self.q_table[(state, a)]
                                for a in range(self.n_actions)])
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self):
        """
        decrement_epsilon
        """
        self.epsilon = self.epsilon * self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        """
        learn based on the bellman equation
        """
        actions = np.array([self.q_table[(state_, a)]
                            for a in range(self.n_actions)])
        a_max = np.argmax(actions)

        self.q_table[(state, action)] += self.alpha * \
            (reward + self.gamma * self.q_table[(state_, a_max)] -
             self.q_table[(state, action)])

        self.decrement_epsilon()
