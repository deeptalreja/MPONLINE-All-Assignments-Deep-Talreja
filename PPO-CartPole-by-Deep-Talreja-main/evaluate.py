"""
evaluate.py
Loads the trained PPO agent and evaluates it deterministically over
100 episodes on CartPole-v1, writing a summary report to disk.

Author: Deep Talreja
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_PATH = "ppo_model"
N_EVAL_EPISODES = 100
REPORT_PATH = "evaluation_report.txt"


def evaluate() -> None:
    env = gym.make("CartPole-v1")
    model = PPO.load(MODEL_PATH, env=env)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )

    # CartPole-v1 considers the task "solved" at an average reward of 475+
    solved = mean_reward >= 475.0

    report_lines = [
        "PPO CartPole-v1 Evaluation Report",
        "=" * 40,
        f"Episodes evaluated : {N_EVAL_EPISODES}",
        f"Mean reward        : {mean_reward:.2f}",
        f"Std deviation       : {std_reward:.2f}",
        f"Solved (>= 475)     : {'Yes' if solved else 'No'}",
        "=" * 40,
        "Author: Deep Talreja",
    ]
    report_text = "\n".join(report_lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text + "\n")

    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
