# This code implements and compares different multi-armed bandit algorithms in a non-stationary environment
# (The true values of actions undergo random walks)

# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Environment class to simulate a non-stationary multi-armed bandit problem
#   true value (q*) of each arm does a random walk at each time step
class NonStationaryBandit:
    # initializes bandit environment; sets std of random walk step size
    def __init__(self, k=10, random_walk_std=0.01):
        self.k = k # steps
        self.random_walk_std = random_walk_std # std of step size
        self.reset() # makes sure is reset
    
    # resets the environment (all true values start at same value = 0)
    def reset(self):
        self.q_star = np.zeros(self.k)
    
    # does one step of the random walk for all arms
    def step(self):
        self.q_star += np.random.normal(0, self.random_walk_std, self.k)
    
    # returns reward for selected arm
    def pull(self, arm):
        # reward drawn from normal distribution with mean q*(arm) and unit variance
        return np.random.normal(self.q_star[arm], 1)
    
# Agent implementation for e-greedy action selection strategy
#   Can use either sample-average or constant step-size updates
class EpsilonGreedyAgent:
    # initialize agent with number of arms, epsilon, constant step size flag, and alpha: step size parameter (only used if constant_step_size=True)
    def __init__(self, n_arms, epsilon, constant_step_size=False, alpha=None):
        self.n_arms = n_arms # number of arms
        self.epsilon = epsilon # epsilon for e-greedy action selection
        self.constant_step_size = constant_step_size # whether to use constant step size or not; if false, uses sample-average
        self.alpha = alpha # step size parameter 
        self.reset()
    
    # resets agent's memory
    def reset(self):
        self.q_estimates = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        self.t = 0
    
    # selects action using e-greedy strategy
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_estimates)
    
    # updates value estimates after getting reward
    def update(self, action, reward):
        self.n_pulls[action] += 1
        self.t += 1
        
        if self.constant_step_size:
            # constant step-size update
            self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])
        else:
            # sample-average update
            self.q_estimates[action] += (reward - self.q_estimates[action]) / self.n_pulls[action]

# Agent implementation for UCB (upper conficdence bound) action selections strategy
class UCBAgent:
    # initialize agent
    def __init__(self, n_arms, alpha):
        self.n_arms = n_arms # number of arms
        self.alpha = alpha # step size parameter
        self.reset()
    
    # reset the agent's memory
    def reset(self):
        self.q_estimates = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        self.t = 0
    
    # select an action using UCB
    def select_action(self):
        # handle initial pulls: try each arm once
        if np.any(self.n_pulls == 0):
            return np.random.choice(np.where(self.n_pulls == 0)[0])
        
        # UCB action selection
        ucb_values = self.q_estimates + np.sqrt(2 * np.log(self.t) / self.n_pulls)
        return np.argmax(ucb_values)
    
    # update value estimates after getting reward
    def update(self, action, reward):
        self.n_pulls[action] += 1
        self.t += 1
        # using constant step-size update
        self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])

# run a complete experiment with the specified agent type and parameters
def run_experiment(agent_type, n_steps, n_runs, **agent_params):
    rewards = np.zeros((n_runs, n_steps))
    optimal_actions = np.zeros((n_runs, n_steps))
    
    for run in range(n_runs):
        if run % 100 == 0:  # progress indicator for logging output
            print(f"Starting run {run}/{n_runs}")
            
        bandit = NonStationaryBandit()
        agent = agent_type(**agent_params)
        
        for step in range(n_steps):
            bandit.step()
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
            
            rewards[run, step] = reward
            optimal_actions[run, step] = action == np.argmax(bandit.q_star)
    
    return rewards, optimal_actions

# create comparison plots for the results (mimicking those shown in Figure 2.2 in the Sutton & Barto)
def plot_comparison_results(results_dict, n_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    steps = np.arange(1, n_steps + 1)
    colors = ['blue', 'orange', 'green']
    
    # Plot 1: plot average rewards
    ax1.set_title('Average Reward over Time')
    for (name, results), color in zip(results_dict.items(), colors):
        rewards, _ = results
        mean_reward = np.mean(rewards, axis=0)
        std_reward = np.std(rewards, axis=0) / np.sqrt(rewards.shape[0])
        ax1.plot(steps, mean_reward, label=name, color=color)
        ax1.fill_between(steps, 
                        mean_reward - std_reward, 
                        mean_reward + std_reward, 
                        alpha=0.1, 
                        color=color)
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: plot percentage of optimal actions
    ax2.set_title('Percentage of Optimal Actions over Time')
    for (name, results), color in zip(results_dict.items(), colors):
        _, optimal = results
        mean_optimal = np.mean(optimal, axis=0) * 100
        std_optimal = np.std(optimal, axis=0) * 100 / np.sqrt(optimal.shape[0])
        ax2.plot(steps, mean_optimal, label=name, color=color)
        ax2.fill_between(steps, 
                        mean_optimal - std_optimal, 
                        mean_optimal + std_optimal, 
                        alpha=0.1, 
                        color=color)
    
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal Action')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# run experiment to solve Exercise 2.5: compare sample-average and constant step-size
def run_original_experiment(n_steps, n_runs):
    print("Running sample-average ε-greedy experiment...")
    sample_avg_results = run_experiment(
        EpsilonGreedyAgent,
        n_steps,
        n_runs,
        n_arms=10,
        epsilon=0.1,
        constant_step_size=False
    )
    
    print("Running constant step-size ε-greedy experiment...")
    constant_alpha_results = run_experiment(
        EpsilonGreedyAgent,
        n_steps,
        n_runs,
        n_arms=10,
        epsilon=0.1,
        constant_step_size=True,
        alpha=0.1
    )
    
    results_dict = {
        'Sample Average ε-greedy': sample_avg_results,
        'Constant α ε-greedy': constant_alpha_results
    }
    
    # plot and save results
    fig = plot_comparison_results(results_dict, n_steps)
    plt.savefig('non_stationary_bandit_results.png')
    plt.show()
    
    return results_dict

# run extension of experiment: look at how UCB performs in this nonstationary environment
def run_ucb_experiment(n_steps, n_runs):
    print("Running sample-average ε-greedy experiment...")
    sample_avg_results = run_experiment(
        EpsilonGreedyAgent,
        n_steps,
        n_runs,
        n_arms=10,
        epsilon=0.1,
        constant_step_size=False
    )
    
    print("Running constant step-size ε-greedy experiment...")
    constant_alpha_results = run_experiment(
        EpsilonGreedyAgent,
        n_steps,
        n_runs,
        n_arms=10,
        epsilon=0.1,
        constant_step_size=True,
        alpha=0.1
    )
    
    print("Running UCB experiment...")
    ucb_results = run_experiment(
        UCBAgent,
        n_steps,
        n_runs,
        n_arms=10,
        alpha=0.1
    )
    
    results_dict = {
        'Sample Average ε-greedy': sample_avg_results,
        'Constant α ε-greedy': constant_alpha_results,
        'UCB (α=0.1)': ucb_results
    }
    
    # plot and save results
    fig = plot_comparison_results(results_dict, n_steps)
    plt.savefig('non_stationary_bandit_comparison.png')
    plt.show()
    
    return results_dict

# MAIN function
if __name__ == "__main__":
    # print welcome message and explanation
    print("\nNon-stationary Multi-Armed Bandit Experiment")
    print("===========================================")
    print("\nThis program implements and compares different approaches to the")
    print("multi-armed bandit problem in a non-stationary environment where")
    print("the true values of actions undergo random walks.\n")
    
    # offer/explain available experiments
    print("Available experiments:")
    print("1. Original Comparison (Exercise 2.5 from Sutton & Barto)")
    print("   - Compares sample-average vs constant step-size methods")
    print("   - Both use ε-greedy exploration (ε=0.1)")
    print("\n2. UCB Extension")
    print("   - Adds UCB (Upper Confidence Bound) agent to the comparison")
    print("   - Tests how UCB performs in non-stationary environments")
    print("   - Includes all three methods for direct comparison\n")
    
    # get user input for which experiment to run
    while True:
        choice = input("Enter experiment number (1 or 2): ")
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # set parameters
    n_steps = 10000
    n_runs = 2000
    
    print(f"\nRunning experiment with {n_runs} runs of {n_steps} steps each.")
    print("This may take several minutes...\n")
    
    # run selected experiment
    start_time = time.time()
    if choice == '1':
        results = run_original_experiment(n_steps, n_runs)
    else:
        results = run_ucb_experiment(n_steps, n_runs)
    
    elapsed_time = time.time() - start_time
    print(f"\nExperiment completed in {elapsed_time:.1f} seconds.")
    print("Results have been plotted and saved to file.")