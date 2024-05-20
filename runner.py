# run_simulation.py
from Environment import create_taxi_environment
from agent import TD0Agent, MonteCarloAgent, TD5Agent
import matplotlib.pyplot as plt

def run_simulation_TD0():
    env = create_taxi_environment()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = TD0Agent(n_states, n_actions)

    n_episodes = 1000
    episode_length = 1000
    rewards_per_episode = []

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0

        for _ in range(episode_length):

            while not done and not truncated:
                action = agent.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward


        rewards_per_episode.append(total_reward)

    print(agent.get_q_table())

    return rewards_per_episode

# run_mc_simulation.py
def run_mc_simulation():
    env = create_taxi_environment()
    agent = MonteCarloAgent(env.observation_space.n, env.action_space.n)
    episode_length = 1000
    num_episodes = 100
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        episode_transitions = []

        for _ in range(episode_length):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            episode_transitions.append((state, action, reward))
            state = next_state
            total_reward += reward
            if done:
                break

        agent.update_Q(episode_transitions)

        episode_rewards.append(total_reward)

    return episode_rewards

def run_simulation_TD5():
    env = create_taxi_environment()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = TD5Agent(n_states, n_actions)

    n_episodes = 1000
    episode_length = 1000
    rewards_per_episode = []

    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        transitions = []

        for _ in range(episode_length):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            transitions.append((state, action, reward, next_state))

            if len(transitions) == agent.n_step or done:
                agent.update_q_values(transitions)
                transitions = []

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        agent.reset_eligibility_traces()
        rewards_per_episode.append(total_reward)

    print(agent.get_q_table())

    return rewards_per_episode

def plot_rewards(rewards_per_episode, title):

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Accumulated Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    #td0_rewards = run_simulation_TD0()
    #plot_rewards(td0_rewards, 'TD0 Agent')

    #mc_rewards = run_mc_simulation()
    #plot_rewards(mc_rewards, 'Monte Carlo Agent')
    
    
    td5_rewards = run_simulation_TD5()
    plot_rewards(td5_rewards, 'TD5 Agent')
