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

for /f "delims=" %%i in ('"%PYTHON_EXE%" -c "from train.async_train_universal import detect_active_log_level, build_tensorboard_logdir_spec; import os; b=os.path.join(r'%REPO_ROOT%','logs_universal'); a=detect_active_log_level(b); print(a or '')"') do set "ACTIVE_LEVEL=%%i"

if defined ACTIVE_LEVEL (
  for /f "delims=" %%i in ('"%PYTHON_EXE%" -c "from train.async_train_universal import build_tensorboard_logdir_spec; import os; print(build_tensorboard_logdir_spec(os.path.join(r'%REPO_ROOT%','logs_universal'), r'%ACTIVE_LEVEL%'))"') do set "LOGDIR_SPEC=%%i"
  echo Active level ^(latest logs_universal folder^): %ACTIVE_LEVEL%
) else (
  set "LOGDIR_SPEC=all:%REPO_ROOT%\logs_universal"
  echo No level folders found under logs_universal yet.
)

echo TensorBoard logdir_spec: %LOGDIR_SPEC%
echo Open: http://localhost:6006
echo In the left sidebar enable run "active" for %ACTIVE_LEVEL%, or browse "all" for every level.
echo Training curves appear after the first learner update; async/learner_ready shows immediately.
echo.

"%PYTHON_EXE%" -m tensorboard.main --logdir_spec "%LOGDIR_SPEC%" --port 6006 --reload_interval 15 --load_fast false --bind_all
if /i not "%~1"=="--no-pause" pause
