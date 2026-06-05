@echo off
call venv\Scripts\activate.bat
echo === ENVIRONMENT ACTIVATED ===
echo === STARTING QA PIPELINE ===
python agent.py
echo === QA PIPELINE COMPLETE ===
