@echo off
REM run.bat
REM One-command setup + launch for the Car Price Predictor project.
REM Author: Deep Talreja

echo Step 1/3: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo Step 2/3: Training the model...
python train.py
if errorlevel 1 goto :error

echo Step 3/3: Starting the Flask server on port 5002...
set PORT=5002
python app.py
goto :eof

:error
echo Something went wrong. Check the messages above.
pause
