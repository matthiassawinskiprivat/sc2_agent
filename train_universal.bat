@echo off
setlocal EnableExtensions

echo Starting Universal (Command + Spatial) SC2 Training Pipeline...
echo Using conda env: sc2_agent_cuda

if defined SC2_REPO_ROOT (
  set "REPO_ROOT=%SC2_REPO_ROOT%"
) else (
  set "REPO_ROOT=%~dp0"
)
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

if exist "%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe"
) else if exist "%USERPROFILE%\miniconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\sc2_agent_cuda\python.exe"
) else (
  echo ERROR: sc2_agent_cuda not found.
  pause
  exit /b 1
)

cd /d "%REPO_ROOT%"
set "PYTHONPATH=%REPO_ROOT%"
set SC2_SKIP_KILL_SWITCH=1

"%PYTHON_EXE%" -u "%REPO_ROOT%\train\train_universal.py"
pause
exit /b %ERRORLEVEL%
