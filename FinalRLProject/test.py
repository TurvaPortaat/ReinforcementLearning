import gymnasium as gym

for episode in range():
    #(num_episodes)


    # Creating LunarLander-environment
    env = gym.make("LunarLander-v3", render_mode="human")

    # Reset the environment
    state, _ = env.reset()

# Do 1000 rounds with random actions
    for _ in range(1000):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)

        if done:    # When done restart
            state, _ = env.reset()

        #memory.push(state, action, reward, done)

env.close()

