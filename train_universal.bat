@echo off
setlocal
echo Starting Universal (Command + Spatial) SC2 Training Pipeline...
echo Using conda env: sc2_agent_cuda

where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: conda is not on PATH. Open Anaconda Prompt, run conda init cmd.exe, or add conda to PATH.
  pause
  exit /b 1
)

set PYTHONPATH=%PYTHONPATH%;%CD%
set SC2_SKIP_KILL_SWITCH=1
conda run -n sc2_agent_cuda --no-capture-output python train/train_universal.py
if errorlevel 1 (
  echo.
  echo If the environment is missing, create it from the repo root:
  echo   conda env create -f environment_cuda.yml
  echo Verify GPU after install:
  echo   conda run -n sc2_agent_cuda python scripts/verify_torch_cuda.py
)
pause
