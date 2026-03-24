@echo off
echo Starting ASYNC Universal (AlphaStar-lite) SC2 Training Pipeline...
echo Offloading large observation layers to C:/temp/sc2_async_buffer...
set PYTHONPATH=%PYTHONPATH%;%CD%
set SC2_SKIP_KILL_SWITCH=1
python train/async_train_universal.py
if errorlevel 1 pause
