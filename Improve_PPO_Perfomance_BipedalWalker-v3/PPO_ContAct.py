from stable_baselines3 import PPO
import gymnasium as gym


env = gym.make('BipedalWalker-v3', render_mode='human')

# Train the model
# model = PPO('MlpPolicy', env, verbose=1)  # CnnPlicy
#model = PPO(
#    'MlpPolicy',
#    env,
#    learning_rate=0.0001,
#    n_steps=2028,
#    batch_size=64,
#    verbose=1)
#
#model.learn(total_timesteps=200000)
#model.save('ppo_BipedalWalker')
#
#print("Train is done!")

model = PPO.load('ppo_BipedalWalker')

obs, _ = env.reset()
for _ in range(500):
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
env.close()

