@echo off
echo Starting Original SC2 Training Pipeline...
set PYTHONPATH=%PYTHONPATH%;%CD%
python train/train_pipeline.py
pause
