from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('BipedalWalker-v3')
model = PPO("MlpPolicy", env)
print("Kaikki toimii!")
