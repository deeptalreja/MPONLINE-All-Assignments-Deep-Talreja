"""
test.py
Renders a live visual window showing the trained PPO agent
balancing the pole in the CartPole-v1 environment.

Author: Deep Talreja
"""

import time
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "ppo_model"
N_EPISODES = 5


def watch_agent() -> None:
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load(MODEL_PATH, env=env)

    for episode in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.01)  # slight delay so the window is watchable

        print(f"Episode {episode}: total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    watch_agent()
