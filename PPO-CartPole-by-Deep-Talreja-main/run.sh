#!/usr/bin/env bash
# run.sh
# One-command setup + full pipeline for the PPO CartPole project.
# Author: Deep Talreja

set -e

echo "Step 1/4: Installing dependencies..."
pip install -r requirements.txt

echo "Step 2/4: Training the PPO agent (50,000 timesteps)..."
python train.py

echo "Step 3/4: Evaluating the trained agent over 100 episodes..."
python evaluate.py

echo "Step 4/4: Launching visual test window..."
python test.py
