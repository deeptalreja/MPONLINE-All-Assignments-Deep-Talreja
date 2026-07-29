#!/usr/bin/env bash
# run.sh
# One-command setup + launch for the Car Price Predictor project.
# Author: Deep Talreja

set -e

echo "Step 1/3: Installing dependencies..."
pip install -r requirements.txt

echo "Step 2/3: Training the model..."
python train.py

echo "Step 3/3: Starting the Flask server on port 5002..."
export PORT=5002
python app.py
