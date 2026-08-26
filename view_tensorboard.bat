@echo off
setlocal EnableExtensions
REM TensorBoard for unit-control (logs_universal) AND build agent (logs_build_zerg).

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
  if /i not "%~1"=="--no-pause" pause
  exit /b 1
)

cd /d "%REPO_ROOT%"
set "PYTHONPATH=%REPO_ROOT%"

"%PYTHON_EXE%" -u "%REPO_ROOT%\train\view_tensorboard.py" --repo-root "%REPO_ROOT%"
set "EXIT_CODE=%ERRORLEVEL%"
if /i not "%~1"=="--no-pause" pause
exit /b %EXIT_CODE%
