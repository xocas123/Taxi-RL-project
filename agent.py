# td0_agent.py
import numpy as np
import random

class TD0Agent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if not isinstance(state, int):
            raise ValueError(f"State must be an integer, got {type(state)} instead.")
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])  # Best action for next state
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def get_q_table(self):
        return self.q_table



class MonteCarloAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.returns = {state: {action: [] for action in range(n_actions)} for state in range(n_states)}
        self.policy = np.random.randint(n_actions, size=n_states)

    def choose_action(self, state):
        """ Epsilon-greedy policy """
        if random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self.policy[state]

    def update_policy(self, state):
        """ Greedy policy improvement """
        self.policy[state] = np.argmax(self.Q[state])

    def update_Q(self, episode):
        """ Update action value function Q based on the episode """
        G = 0
        gamma = 0.9  # Discount factor can be adjusted
        for state, action, reward in reversed(episode):
            G = reward + gamma * G  # Update total return
            # Append G to returns if it's the first visit to the state-action pair
            if not (state, action) in [(x, y, r) for x, y, r in episode[:-1]]:
                self.returns[state][action].append(G)
                self.Q[state][action] = np.mean(self.returns[state][action])
                self.update_policy(state)
