@echo off
echo Starting Universal (Command + Spatial) SC2 Training Pipeline...
set PYTHONPATH=%PYTHONPATH%;%CD%
set SC2_SKIP_KILL_SWITCH=1
python train/train_universal.py
pause
