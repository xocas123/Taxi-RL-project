# td0_agent.py
import numpy as np
import random


class QLearningAgent:

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma=1):


        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state):

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q(self, state, action, reward, next_state):

        self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])


class TD0Agent:
    def __init__(self, n_states, n_actions, alpha, epsilon, gamma=1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])  # Best action for next state
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error


class MonteCarloAgent:
    def __init__(self, n_states, n_actions, epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.returns = {state: {action: [] for action in range(n_actions)} for state in range(n_states)}
        self.policy = np.random.randint(n_actions, size=n_states)

    def select_action(self, state):
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


class TD5Agent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.eligibility_traces = np.zeros((n_states, n_actions))
        self.n_step = 5  # Define the number of steps for TD(n)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_values(self, transitions):
        """ Aktualisieren  Q-Werte. Falta arreglar,"""
        for t, (state, action, reward, next_state) in enumerate(transitions):
            self.eligibility_traces[state][action] += 1

            n_step_return = sum([self.gamma ** i * transitions[i][2] for i in range(len(transitions))])
            if len(transitions) == self.n_step:
                n_step_return += self.gamma ** self.n_step * np.max(self.q_table[next_state])

            # Update qval
            td_error = n_step_return - self.q_table[state][action]
            self.q_table += self.alpha * td_error * self.eligibility_traces

            # Decay eligibility traces
            self.eligibility_traces *= self.gamma * 0.9

    def reset_eligibility_traces(self):
        self.eligibility_traces = np.zeros((self.n_states, self.n_actions))

    def get_q_table(self):
        return self.q_table


class TDnAgent:
    def __init__(self, n_states, n_actions, n=1, alpha=0.1, gamma=0.99, epsilon=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.eligibility_traces = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_values(self, transitions):
        """ Update Q-values based on n-step transitions """
        for t, (state, action, reward, next_state) in enumerate(transitions):
            self.eligibility_traces[state][action] += 1

            # n pasos, deveulta
            n_step_return = sum([self.gamma ** i * transitions[i][2] for i in range(len(transitions))])
            if len(transitions) == self.n:
                n_step_return += self.gamma ** self.n * np.max(self.q_table[next_state])

            # qval & eligib.
            td_error = n_step_return - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error * self.eligibility_traces[state][action]

            # decay
            self.eligibility_traces *= self.gamma * 0.9

    def reset_eligibility_traces(self):
        self.eligibility_traces = np.zeros((self.n_states, self.n_actions))

    def get_q_table(self):
        return self.q_table
