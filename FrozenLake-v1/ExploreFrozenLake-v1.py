import gymnasium as gym

from test_frozenlake import action, new_obs, reward

# Creating and resetting the environment
env = gym.make("FrozenLake-v1", is_slippery=True)
obs, info = env.reset()

# Printing environment's observation and action spaces
print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)

# Running 5 random actions and printing them, also adding the rewards
for step in range(5):
    action = env.action_space.sample()
    new_obs, reward, done, truncated, info = env.step(action)
    print(f"Step {step+1}: Obs={new_obs}, Reward={reward}, Done={done}, Truncated={truncated}")