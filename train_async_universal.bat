@echo off
setlocal EnableExtensions

echo Starting ASYNC Universal (AlphaStar-lite) SC2 Training Pipeline...
echo Using conda env: sc2_agent_cuda

if defined SC2_REPO_ROOT (
  set "REPO_ROOT=%SC2_REPO_ROOT%"
) else (
  set "REPO_ROOT=%~dp0"
)
REM Remove trailing backslash (avoids broken quoting on some tools).
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

if defined SC2_ASYNC_BUFFER_DIR (
  echo Async buffer: %SC2_ASYNC_BUFFER_DIR%
) else (
  echo Async buffer default: C:\temp\sc2_async_buffer
)
echo TensorBoard: http://localhost:6006  (restarts each curriculum level)
echo Repo: %REPO_ROOT%

if exist "%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe"
) else if exist "%USERPROFILE%\miniconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\sc2_agent_cuda\python.exe"
) else if exist "C:\ProgramData\anaconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\sc2_agent_cuda\python.exe"
) else (
  echo ERROR: sc2_agent_cuda not found. Create it with:
  echo   conda env create -f "%REPO_ROOT%\environment_cuda.yml"
  pause
  exit /b 1
)

if not exist "%REPO_ROOT%\train\async_train_universal.py" (
  echo ERROR: "%REPO_ROOT%\train\async_train_universal.py" not found.
  pause
  exit /b 1
)

cd /d "%REPO_ROOT%"
set "PYTHONPATH=%REPO_ROOT%"
set SC2_SKIP_KILL_SWITCH=1

echo Python: %PYTHON_EXE%
echo Working directory: %CD%

"%PYTHON_EXE%" -u "%REPO_ROOT%\train\async_train_universal.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Training exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
