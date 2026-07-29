"""
train.py
Trains a PPO (Proximal Policy Optimization) agent on the Gymnasium
CartPole-v1 environment using Stable-Baselines3, then saves the
trained policy and a learning curve plot.

Author: Deep Talreja
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

LOG_DIR = "./logs"
MODEL_PATH = "ppo_model"
TOTAL_TIMESTEPS = 50_000


def train() -> None:
    import os
    os.makedirs(LOG_DIR, exist_ok=True)

    # Wrap the environment with Monitor so episode rewards get logged to disk
    env = gym.make("CartPole-v1")
    env = Monitor(env, LOG_DIR)

    model = PPO("MlpPolicy", env, verbose=1)

    print(f"Training PPO agent for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")

    plot_learning_curve()


def plot_learning_curve() -> None:
    """Reads the Monitor logs and plots episode reward vs. timesteps."""
    x, y = ts2xy(load_results(LOG_DIR), "timesteps")

    if len(x) == 0:
        print("No episodes logged yet — skipping learning curve plot.")
        return

    # Smooth the reward curve with a simple moving average for readability
    window = min(50, len(y))
    if window > 1:
        y_smoothed = np.convolve(y, np.ones(window) / window, mode="valid")
        x_smoothed = x[window - 1:]
    else:
        y_smoothed, x_smoothed = y, x

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, alpha=0.3, color="#2d6a4f", label="Episode reward")
    plt.plot(x_smoothed, y_smoothed, color="#1b4332", linewidth=2,
              label=f"Moving average ({window} episodes)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("PPO Training Progress — CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    print("Learning curve saved to learning_curve.png")


if __name__ == "__main__":
    train()
