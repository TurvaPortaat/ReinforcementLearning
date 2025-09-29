from stable_baselines3 import PPO

import gymnasium as gym


# Creating and resetting the environment
env = gym.make("BipedalWalker-v3")
obs, info = env.reset()
