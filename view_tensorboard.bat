@echo off
setlocal EnableExtensions
REM TensorBoard for universal logs. Frees port 6006 first, then starts one server.

if defined SC2_REPO_ROOT (
  set "REPO_ROOT=%SC2_REPO_ROOT%"
) else (
  set "REPO_ROOT=%~dp0"
)
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

if exist "%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe"
) else (
  echo ERROR: sc2_agent_cuda not found.
  if /i not "%~1"=="--no-pause" pause
  exit /b 1
)

cd /d "%REPO_ROOT%"
set "PYTHONPATH=%REPO_ROOT%"

echo Freeing port 6006 if needed...
"%PYTHON_EXE%" -c "from train.async_train_universal import free_port; free_port(6006)"

echo TensorBoard logdir: %REPO_ROOT%\logs_universal
echo Open: http://localhost:6006
echo Runs: level_attack_0, level_attack_1, level_move_single_0, ...
echo Tip: enable ONE run in the left sidebar; refresh the page after a new level starts.
echo.

"%PYTHON_EXE%" -m tensorboard.main --logdir "%REPO_ROOT%\logs_universal" --port 6006 --reload_interval 5 --bind_all
if /i not "%~1"=="--no-pause" pause
