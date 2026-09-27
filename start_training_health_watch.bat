@echo off
setlocal EnableExtensions
REM Copy this file to your home directory and double-click / run it to start
REM the long-running training health watcher (checks about once per hour).
REM
REM Requirements:
REM   1. Set SC2_REPO_ROOT below (or as a user environment variable)
REM   2. conda env sc2_agent_cuda
REM   3. Optional LLM: install Cursor agent CLI and put `agent` on PATH
REM      (or set CURSOR_AGENT_BIN). Without it, deterministic digests still run.
REM
REM Cost defaults:
REM   - Model: composer-2.5-fast (override with SC2_HEALTH_MODEL or --model)
REM   - LLM only runs on WARN/CRIT (use --llm-always to force every hour)
REM   - LLM sees digest + alerts_excerpt only (not full logs)

REM ---- configure ---------------------------------------------------------
if not defined SC2_REPO_ROOT set "SC2_REPO_ROOT=D:\sc2_unit_agent"
REM Optional: full path to Cursor agent CLI
REM set "CURSOR_AGENT_BIN=%USERPROFILE%\.local\bin\agent.exe"
REM Optional: override cheap triage model
REM set "SC2_HEALTH_MODEL=composer-2.5-fast"
REM ------------------------------------------------------------------------

echo Starting SC2 training health watcher...
echo Repo: %SC2_REPO_ROOT%

if not exist "%SC2_REPO_ROOT%\train\training_health_watch.py" (
  echo ERROR: "%SC2_REPO_ROOT%\train\training_health_watch.py" not found.
  echo Edit SC2_REPO_ROOT at the top of this bat file.
  pause
  exit /b 1
)

if exist "%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\sc2_agent_cuda\python.exe"
) else if exist "%USERPROFILE%\miniconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\sc2_agent_cuda\python.exe"
) else if exist "C:\ProgramData\anaconda3\envs\sc2_agent_cuda\python.exe" (
  set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\sc2_agent_cuda\python.exe"
) else (
  echo ERROR: sc2_agent_cuda not found. Create it with:
  echo   conda env create -f "%SC2_REPO_ROOT%\environment_cuda.yml"
  pause
  exit /b 1
)

cd /d "%SC2_REPO_ROOT%"
set "PYTHONPATH=%SC2_REPO_ROOT%"

echo Python: %PYTHON_EXE%
echo Reports: %SC2_REPO_ROOT%\logs_universal\_health\  ^(latest.md only; cleaned on start^)
echo Alerts log: %SC2_REPO_ROOT%\logs\process_alerts.log  ^(truncated on watcher start^)
echo Interval: 3600s  ^(Ctrl+C to stop^)
echo Model default: composer-2.5-fast  ^(LLM only on WARN/CRIT^)
echo.

"%PYTHON_EXE%" -u "%SC2_REPO_ROOT%\train\training_health_watch.py" --repo-root "%SC2_REPO_ROOT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Watcher exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
