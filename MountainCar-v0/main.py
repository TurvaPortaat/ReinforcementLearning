import gymnasium as gym

# Creating and resetting the environment
env = gym.make("MountainCar-v0")
obs, info = env.reset()

# Printing the observation and action spaces
print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)

# Running a short loop (5 steps) with random action
for step in range(5):
    action = env.action_space.sample() # Pick a random action
    new_obs, reward, done, truncated, info = env.step(action) # Take action in environment
    print(f"Step {step+1}: Obs={new_obs}, Reward={reward}, Done={done}, Truncated={truncated}")

