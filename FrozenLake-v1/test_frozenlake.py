import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
obs, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for step in range(5):
    action = env.action_space.sample()
    new_obs, reward, done, truncated, info = env.step(action)
    print(f"Step {step+1}: Obs={new_obs}, Reward={reward}, Done={done}, Truncated={truncated}")
    if done or truncated:
        env.reset()
