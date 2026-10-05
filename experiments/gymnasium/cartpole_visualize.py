# This file accompanies the ipynb for visualizing the environment in a human-readable format
import gymnasium as gym
import lantern.robot as lanbot
import matplotlib.pyplot as plt

# Create environment
env = gym.make("CartPole-v1", max_episode_steps=100000, render_mode="human")

# Reset environment 
observation, info = env.reset()
# observation: what the agent can "see" (cart position, velocity, pole angle, etc.)
# info: extra debugging info
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an aciton (0 = push cart left, 1 = push cart right)
    action = env.action_space.sample() # This is a random action for now (policy is to pick a random action with equal probs)
    
    # Take action and observe what happens
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Add reward to cumulative reward
    total_reward += reward
    episode_over = terminated or truncated
    
print(f"Episode terminated! Total reward: {total_reward}")
env.close()