@echo off
echo Starting Combined TensorBoard...
echo Open http://localhost:6006 in your browser.
tensorboard --logdir_spec "original:logs,universal:logs_universal" --port 6006
pause
