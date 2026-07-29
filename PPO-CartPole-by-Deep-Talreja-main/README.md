# PPO CartPole-v1 Reinforcement Learning

**Submitted By:** Deep Talreja
**Application Number:** IN26010914
**Registration No:** 23BCE11003
**Internship:** MP Online AI/ML Internship

This project implements the Proximal Policy Optimization (PPO) algorithm to solve the Gymnasium `CartPole-v1` environment using Stable-Baselines3.

---

## Project Structure

- `train.py` — Initializes the environment, trains the PPO agent for 50,000 timesteps, and saves training logs + a learning curve plot.
- `evaluate.py` — Evaluates the trained model over 100 deterministic test episodes and writes a summary report.
- `test.py` — Renders a live visual window showing the trained model balancing the pole.
- `ppo_model.zip` — Saved policy network weights (generated after training).
- `learning_curve.png` — Training performance graph showing reward progression (generated after training).
- `evaluation_report.txt` — Text report of evaluation results (generated after evaluation).

---

## Installation & Setup

1. **Create and activate a clean Conda environment:**

```bash
conda create -n cartpole-ppo python=3.10 -y
conda activate cartpole-ppo
```

2. **Install the required packages:**

```bash
pip install stable-baselines3 gymnasium matplotlib
```

---

## How to Run

### Quick Start (Automated)

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```bat
run.bat
```

This runs the full pipeline — install, train, evaluate, and visually test — in one go.

### Manual, Step-by-Step

#### 1. Train the Agent

Runs the PPO agent for 50,000 timesteps, saves the model checkpoint, and generates the training progress chart.

```bash
python train.py
```

#### 2. Evaluate Performance

Tests the saved checkpoint over 100 separate episodes and writes the performance numbers into a clean text report.

```bash
python evaluate.py
```

#### 3. Visual Testing

Launches a live visual rendering window on your screen to watch the trained model in action.

```bash
python test.py
```

---

## About PPO & CartPole-v1

**CartPole-v1** is a classic control task: balance a pole on a moving cart by pushing it left or right. An episode is considered fully successful once the agent sustains an average reward of 475+ over 100 evaluation episodes (the environment caps episodes at 500 timesteps).

**Proximal Policy Optimization (PPO)** is a policy-gradient reinforcement learning algorithm that improves training stability by limiting how far each policy update can drift from the previous one, avoiding destructively large updates that plain policy-gradient methods are prone to.

---

## Notes

- Training on CPU is sufficient for this environment — no GPU required.
- The `learning_curve.png` plot uses a moving average over episodes to smooth out the natural variance in reward signal during training.
- If you re-run `train.py`, it will overwrite the previous `ppo_model.zip`, `learning_curve.png`, and training logs.
