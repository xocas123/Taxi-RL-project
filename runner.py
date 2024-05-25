# run_simulation.py
from Environment import create_taxi_environment
from agent import TD0Agent, MonteCarloAgent, TD5Agent,TDnAgent
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials

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
def run_mc_simulation(epsilon=0.1):
    env = create_taxi_environment()
    agent = MonteCarloAgent(env.observation_space.n, env.action_space.n, epsilon=epsilon)
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
            if done or truncated:
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

def run_simulation_TDn(n):
    env = create_taxi_environment()
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = TDnAgent(n_states, n_actions, n=n)

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

            if len(transitions) == agent.n or done:
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
    
def optimize_td_hyperparameters(n, max_evals=20):
    def objective(params):
        alpha, gamma, epsilon = params['alpha'], params['gamma'], params['epsilon']
        env = create_taxi_environment()
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        agent = TDnAgent(n_states, n_actions, n=n, alpha=alpha, gamma=gamma, epsilon=epsilon)

        n_episodes = 100
        episode_length = 1000
        total_reward = 0

        for episode in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0
            transitions = []

            for _ in range(episode_length):
                action = agent.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                transitions.append((state, action, reward, next_state))

                if len(transitions) == agent.n or done:
                    agent.update_q_values(transitions)
                    transitions = []

                state = next_state
                episode_reward += reward

                if done or truncated:
                    break

            agent.reset_eligibility_traces()
            total_reward += episode_reward

        avg_reward = total_reward / n_episodes
        return -avg_reward

    space = {
        'alpha': hp.uniform('alpha', 0.01, 1.0),
        'gamma': hp.uniform('gamma', 0.9, 0.999),
        'epsilon': hp.uniform('epsilon', 0.01, 0.1)
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    print("Best hyperparameters found: Alpha={}, Gamma={}, Epsilon={}".format(
        best['alpha'], best['gamma'], best['epsilon']))

    return best

def optimize_mc_hyperparameters(max_evals=20):
    def objective(params):
        epsilon = params['epsilon']
        total_reward = sum(run_mc_simulation(epsilon=epsilon))
        avg_reward = total_reward / 100
        return -avg_reward  # Minimize negative reward to maximize reward

    space = {
        'epsilon': hp.uniform('epsilon', 0.01, 0.1)
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    print("Best epsilon found: epsilon={}".format(best['epsilon']))

    return best
    



if __name__ == "__main__":
    #td0_rewards = run_simulation_TD0()
    #plot_rewards(td0_rewards, 'TD0 Agent')

    #mc_rewards = run_mc_simulation()
    #plot_rewards(mc_rewards, 'Monte Carlo Agent')
    
    
    #td5_rewards = run_simulation_TD5()
    #plot_rewards(td5_rewards, 'TD5 Agent')
    
    #best_result = optimize_td_hyperparameters(n=250, max_evals=20)
    
    best_mc_hyperparameters = optimize_mc_hyperparameters()
