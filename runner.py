# run_simulation.py
from scipy.signal import savgol_filter

from Environment import create_taxi_environment
from agent import TD0Agent, MonteCarloAgent, QLearningAgent
from helper import LearningCurvePlot


def run_simulation_TD0(alpha, epsilon):

    env = create_taxi_environment()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = TD0Agent(n_states, n_actions, alpha, epsilon)

    n_episodes = 10000
    episode_length = 100
    rewards_per_episode = []

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0

        for _ in range(episode_length):

            while not done and not truncated:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward


        rewards_per_episode.append(total_reward)


    return rewards_per_episode

def run_mc_simulation(epsilon):
    env = create_taxi_environment()
    agent = MonteCarloAgent(env.observation_space.n, env.action_space.n, epsilon)
    num_episodes = 10000
    episode_length = 100

    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        episode_transitions = []

        for _ in range(episode_length):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            episode_transitions.append((state, action, reward))
            state = next_state
            total_reward += reward
            if done:
                break

        agent.update_Q(episode_transitions)

        episode_rewards.append(total_reward)

    return episode_rewards


def run_QLearning_simulation(alpha, epsilon):


    env = create_taxi_environment()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(n_actions, n_states, epsilon, alpha)
    print(f"Alpha is {alpha}")

    n_episodes = 10000
    episode_length = 100
    rewards_per_episode = []

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0

        for _ in range(episode_length):

            while not done and not truncated:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.update_q(state, action, reward, next_state)
                state = next_state
                total_reward += reward

        rewards_per_episode.append(total_reward)

    return rewards_per_episode



def plot(reward_sets, labels, title, filename):
    plot = LearningCurvePlot(title=title)
    for rewards, label in zip(reward_sets, labels):
        x = range(len(rewards))
        smoothed_rewards = smooth(rewards, window=51, poly=2)  # Adjust window and poly as needed
        plot.add_curve(x, smoothed_rewards, label=label)
    plot.save(name=filename)



def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)


def run_td0_epsilon_comparisons():
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
    alpha = 0.05

    all_rewards = []
    labels = []

    for epsilon in epsilons:
        td0_rewards = run_simulation_TD0(alpha, epsilon)
        all_rewards.append(td0_rewards)
        labels.append(f"Epsilon {epsilon}")

    plot(all_rewards, labels, "TD0 Agent Performance with Different Epsilons",
                         "td0_epsilon_comparison.png")


def run_td0_alpha_comparisons():
    alphas = [0.01, 0.1, 0.2, 0.5]
    epsilon = 0.05

    all_rewards = []
    labels = []

    for alpha in alphas:
        td0_rewards = run_simulation_TD0(alpha, epsilon)
        all_rewards.append(td0_rewards)
        labels.append(f"Alphas {alpha}")

    plot(all_rewards, labels, "TD0 Agent Performance with Different Alphas", "td0_alpha_comparison.png")


def run_mc_epsilon_comparisons():

    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]

    all_rewards = []
    labels = []

    for epsilon in epsilons:
        mc_rewards = run_mc_simulation(epsilon)
        all_rewards.append(mc_rewards)
        labels.append(f"Epsilon {epsilon}")

    plot(all_rewards, labels, "MC Agent Performance with Different Epsilons", "mc_epsilon_comparison.png")

def run_qlearning_epsilon_comparisons():

    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
    alpha = 0.05

    all_rewards = []
    labels = []

    for epsilon in epsilons:
        qlearning_rewards = run_QLearning_simulation(alpha, epsilon)
        all_rewards.append(qlearning_rewards)
        labels.append(f"Epsilon {epsilon}")

    plot(all_rewards, labels, "QLearning Agent Performance with Different Epsilons","qlearning_epsilon_comparison.png")


def run_qlearning_alpha_comparisons():

    alphas = [0.01, 0.1, 0.2, 0.5]
    epsilon = 0.05

    all_rewards = []
    labels = []

    for alpha in alphas:
        qlearning_rewards = run_QLearning_simulation(alpha, epsilon)
        all_rewards.append(qlearning_rewards)
        labels.append(f"Alpha {alpha}")

    plot(all_rewards, labels, "QLearning Agent Performance with Different Alphas","qlearning_alpha_comparison.png")




if __name__ == "__main__":

    #run_td0_epsilon_comparisons()
    #run_td0_alpha_comparisons()

    #run_mc_epsilon_comparisons()

    #run_qlearning_epsilon_comparisons()
    run_qlearning_alpha_comparisons()

