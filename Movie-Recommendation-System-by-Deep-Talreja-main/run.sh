#!/usr/bin/env bash
# run.sh
# One-command setup + launch for the Movie Recommendation System.
# Author: Deep Talreja

set -e

echo "Step 1/2: Installing dependencies..."
pip install -r requirements.txt

echo "Step 2/2: Starting the Flask server on port 5004..."
export PORT=5004
python app.py
