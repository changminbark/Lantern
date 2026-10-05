# This file accompanies the ipynb for visualizing the environment in a human-readable format
import gymnasium as gym
import lantern.robot as lanbot
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class CartPoleAgent():
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay:float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        
        # Q-tables: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states with the right num of columns (actions)
        # For the CartPole problem, we have the following:
        #   state: (cart_position, cart_velocity, pole_angle, and pole_angular_velocity)
        #   action: 0 = push cart left, 1 = push cart right
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.lr = learning_rate
        self.discount_factor = discount_factor # how much we discount future rewards (multiplicative)
        
        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        # Track training progress
        self.training_error = []
        
        # Discretization
        self.OBS_BOUNDS = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.5, 3.5)]
        self.N_BINS = [6, 6, 12, 12]
        
    def get_action(self, obs: tuple[int, int, int, int]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (left) or 1 (right)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # With probability 1-epsilon: exploitation (best known action, which is the column index)
        return int(np.argmax(self.q_values[obs])) # caching this could also help since this is an O(actions) operation
    
    def update(
        self,
        obs: tuple[int, int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, int, int],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # Get current Q-value
        current_q_value = self.q_values[obs][action]
        
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # What should the Q-value for the current state be (Bellman optimality equation)
        target = reward + self.discount_factor * future_q_value
        
        # How wrong was our current estimate?
        temporal_difference = target - current_q_value
        
        # Update our current estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            current_q_value + self.lr * temporal_difference
        )
        
        # Track learning progress
        self.training_error.append(temporal_difference)
        
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
        
def discretize(obs_bounds, n_bins, obs) -> tuple[int, int, int, int]:
    """Converts continuous values into discrete ones via bucketing"""
    bin_edges = [np.linspace(lo, hi, n+1)[1:-1] for (lo, hi), n in zip(obs_bounds, n_bins)]
    return tuple(int(np.digitize(x, edges)) for x, edges in zip(obs, bin_edges))
        
        
# Training hyperparameters
OBS_BOUNDS = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.5, 3.5)]
N_BINS = [6, 6, 12, 12]
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

# Create environment
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Create training agent
agent = CartPoleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Train
for episode in tqdm(range(n_episodes)):
    # Start a new episode
    obs, info = env.reset()
    done = False
    
    # Finish episode (pole falls or time limit)
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(discretize(OBS_BOUNDS, N_BINS, obs))
        
        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Update current estimate
        agent.update(discretize(OBS_BOUNDS, N_BINS, obs), action, reward, terminated, discretize(OBS_BOUNDS, N_BINS, next_obs))
        
        # Move to next state
        done = terminated or truncated
        obs = next_obs
        
    # Reduce exploration rate
    agent.decay_epsilon()
env.close()
    
# Visualize trained policy
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="runs/gymnasium/cartpole",
    name_prefix="trained",
    episode_trigger=lambda ep: True,  # record every episode
)
agent.epsilon = 0

for ep in range(3):
    # Reset environment 
    observation, info = env.reset()
    print(f"Starting observation: {observation}")
    episode_over = False
    total_reward = 0

    while not episode_over:
        # Choose an aciton (0 = push cart left, 1 = push cart right)
        action = agent.get_action(discretize(OBS_BOUNDS, N_BINS, observation)) # This is a random action for now (policy is to pick a random action with equal probs)
        
        # Take action and observe what happens
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Add reward to cumulative reward
        total_reward += reward
        episode_over = terminated or truncated
        
    print(f"Episode {ep} terminated! Total reward: {total_reward}")
    
env.close()