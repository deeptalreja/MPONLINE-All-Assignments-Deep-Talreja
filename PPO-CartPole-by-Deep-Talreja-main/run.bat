@echo off
REM run.bat
REM One-command setup + full pipeline for the PPO CartPole project.
REM Author: Deep Talreja

echo Step 1/4: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo Step 2/4: Training the PPO agent (50,000 timesteps)...
python train.py
if errorlevel 1 goto :error

echo Step 3/4: Evaluating the trained agent over 100 episodes...
python evaluate.py
if errorlevel 1 goto :error

echo Step 4/4: Launching visual test window...
python test.py
goto :eof

:error
echo Something went wrong. Check the messages above.
pause
