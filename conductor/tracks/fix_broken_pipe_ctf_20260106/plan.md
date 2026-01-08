# Plan: Fix Broken Pipe/EOFError in CTF Training Pipeline

## Objective
Resolve the `BrokenPipeError` / `EOFError` occurring in `train_pipeline.py` during `env.reset()` or `model.learn()`. The error suggests a worker process crash.

## Diagnosis
- [ ] Inspect `logs/worker_*.log` for the specific error that caused the crash.
- [ ] Determine if the crash is related to the recent `CaptureTheFlagSingleEnv` changes (termination logic, `level_name`).
- [ ] Verify if the `level_name="capture_the_flag_0"` change is correctly propagating and if the configuration exists.

## Potential Causes
1.  **Config Mismatch:** `capture_the_flag_0.yaml` might be missing or malformed, causing an unhandled exception in the worker.
2.  **Termination Logic Bug:** The new `_async_step` or `_has_combat_units` logic might have a bug that only triggers in the full pipeline (e.g., race condition, `None` access).
3.  **SC2 Process Crash:** The SC2 client might be crashing due to invalid commands or state.

## Steps
1.  **Analyze Logs:** Read the latest worker logs to pinpoint the exception.
2.  **Reproduce:** Create a minimal reproduction script if the logs point to a specific logic error.
3.  **Fix:** Apply the fix to `CaptureTheFlagSingleEnv` or the configuration.
4.  **Verify:** Run the pipeline or the reproduction script to ensure stability.
