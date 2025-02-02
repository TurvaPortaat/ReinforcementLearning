import numpy as np
import gymnasium as gym
import pickle
import time

# Function to train Q-Learning on a given map
def train_q_learning(env_name, episodes=2000, learning_rate=0.8, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
    env = gym.make("FrozenLake-v1", map_name=env_name, is_slippery=True)  # Keep slippery mode enabled

    # Check if observation space is discrete
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError("Environment's observation space must be discrete.")

    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # Ensure Q-table size matches environment

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        step_count = 0  # Initialize step counter
        max_steps = 200  # Limit the number of steps per episode

        while not done and step_count < max_steps:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, _, _ = env.step(action)

            # Ensure float type for reward calculations
            q_table[state, action] = q_table[state, action] + float(learning_rate) * (
                float(reward) + float(discount_factor) * np.max(q_table[next_state, :]) - q_table[state, action]
            )

            state = next_state
            step_count += 1

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Save progress every 1000 episodes
        if episode % 1000 == 0:
            with open(f"q_table_{env_name}.pkl", "wb") as file_handle:
                pickle.dump(q_table, file_handle)
            print(f"Saved Q-table for {env_name} at episode {episode}")

    return q_table

# Train and save Q-table for 4x4 map
q_table_4x4 = train_q_learning("4x4")
with open("q_table_4x4.pkl", "wb") as file_handle:
    pickle.dump(q_table_4x4, file_handle)

# Train and save Q-table for 8x8 map
q_table_8x8 = train_q_learning("8x8")
with open("q_table_8x8.pkl", "wb") as file_handle:
    pickle.dump(q_table_8x8, file_handle)

print("Training completed and Q-tables saved.")

# Function to evaluate Q-table on a given map
def evaluate(env_name, q_table, step_delay=0.3):
    max_steps = 200  # Prevent infinite loops
    step_count = 0
    env = gym.make("FrozenLake-v1", map_name=env_name, is_slippery=True, render_mode="human")
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done and step_count < max_steps:
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        time.sleep(step_delay)  # Slow down steps for better visualization
        state = next_state
        step_count += 1

    print(f"Total reward on {env_name} map: {total_reward}, Steps Taken: {step_count}")
    env.close()

# Load Q-tables
with open("q_table_4x4.pkl", "rb") as file_handle:
    q_table_4x4 = pickle.load(file_handle)

with open("q_table_8x8.pkl", "rb") as file_handle:
    q_table_8x8 = pickle.load(file_handle)

# Evaluate both maps with their corresponding Q-table
evaluate("4x4", q_table_4x4, step_delay=0.5)  # Slow down the smaller map more
evaluate("8x8", q_table_8x8, step_delay=0.3)
