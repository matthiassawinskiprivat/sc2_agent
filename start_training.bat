@echo off
REM Same launcher as %USERPROFILE%\start_training.bat — delegates to repo async script.
set "SC2_REPO_ROOT=%~dp0"
call "%~dp0train_async_universal.bat"
