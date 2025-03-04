# Non-stationary Bandits

This repository implements and analyzes different approaches to solving non-stationary multi-armed bandit problems. It includes implementations of ε-greedy algorithms with both sample-average and constant step-size updates, as well as an Upper Confidence Bound (UCB) approach. The project focuses on comparing these methods in environments where the true values of actions undergo random walks.

## Installation

1) Clone the repository:
```
git clone https://github.com/kmarr21/nonstationary-bandits.git
cd nonstationary-bandits
```

2) Make sure you have the required packages installed:
```
pip install numpy matplotlib seaborn
```
## Usage

To run the main script:
```
python nonstationary_bandit.py
```
The program will then present you with two experiment options:

1. Original comparison (from Sutton & Barto Exercise 2.5)
   - Compares sample-average vs constant step-size methods
    - Both use ε-greedy exploration (ε=0.1)
2. UCB extension
   - Adds UCB (upper confidence bound) agent to the comparison
    - Tests how UCB performs in non-stationary environments like this one
    - Includes all 3 methods in output plots for direct comparison

Follow the prompts to select your desired experiment. The program will run the simulation and generate plots showing:
- Average reward over time
- Percentage of optimal actions over time

Results will be saved as PNG files in your working directory.

## How It Works
### Environment
The non-stationary bandit environment is implemented in the NonStationaryBandit class:
- Contains k arms (default=10)
- Each arm's true value (q*) starts at 0
- Values undergo random walks with steps drawn from N(0, 0.01)
- Rewards are drawn from N(q*(a), 1) for chosen action a

### Agents
**EpsilonGreedyAgent**:
- Uses ε-greedy action selection (ε=0.1)
- Can use either sample-average or constant step-size updates
- For constant step-size, uses α=0.1

**UCBAgent**:
- Uses UCB1 algorithm for action selection
- Combines estimates with exploration bonus: Q(a) + sqrt(2ln(t)/N(a))
- Uses constant step-size updates (α=0.1)

### Experiments
The code runs multiple independent trials (default=2000) for each agent type, with each trial running for multiple steps (default=10000). It tracks:
1. Rewards received
2. Whether each action was optimal
3. Average performance across all trials

Results are visualized using matplotlib, showing both the mean performance and standard error bands.

## Results
The experiments demonstrate several key findings:

1) In non-stationary environments:
    - Sample-average methods perform poorly due to declining learning rate
    - Constant step-size methods perform better since they can maintain the ability to adapt as the environment changes
    - UCB achieves similar rewards to constant step-size ε-greedy despite selecting optimal actions less frequently
2) The performance gap between methods grows over time, which highlights the importance of maintaining adaptability in non-stationary environments

## Extension: Parameter Study (WIP)

Currently, on the branch titled parameter_study, I am attempting to expand this code to perform a parameter study of different algorithms and their hyperparameters. However, this is a work in progress and I am waiting on more computing power before completing it.  

## Acknowledgments

- Based on Exercise 2.5 from "Reinforcement Learning: An Introduction" by Sutton & Barto
- Extended with UCB implementation for comparison in non-stationary environments
