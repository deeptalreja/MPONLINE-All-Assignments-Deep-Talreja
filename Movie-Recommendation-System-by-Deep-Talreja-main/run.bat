@echo off
REM run.bat
REM One-command setup + launch for the Movie Recommendation System.
REM Author: Deep Talreja

echo Step 1/2: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo Step 2/2: Starting the Flask server on port 5004...
set PORT=5004
python app.py
goto :eof

:error
echo Something went wrong. Check the messages above.
pause
