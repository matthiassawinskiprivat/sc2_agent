# Track Plan: Fix and Stabilize Multi-Processing Training Pipeline

## Phase 1: Analysis & Reproduction
*   [ ] Task: Run existing training script to capture crash logs.
    *   *Goal:* obtain the exact stack trace and error behavior.
    *   *Context:* `debug_train.py` or `train/train_pipeline.py`.
*   [ ] Task: Analyze `train/train_pipeline.py` and Environment structure.
    *   *Goal:* Identify non-picklable objects and windows-specific multiprocessing violations.
    *   *Context:* Check for lambda functions, open file handles, or complex SC2 API objects stored in the Env.

## Phase 2: Core Refactoring
*   [ ] Task: Create a Minimal Reproducible Environment Wrapper.
    *   *Goal:* Create a simplified version of the environment to test basic parallel launching without full game logic, if needed, or isolate the culprit.
    *   *Context:* `tests/test_mp_isolation.py`.
*   [ ] Task: Refactor `level_0_env.py` for Serialization.
    *   *Goal:* Ensure `__init__` and state handling are safe for `spawn`.
*   [ ] Task: Implement/Fix `SubprocVecEnv` instantiation.
    *   *Goal:* Use correct start methods (`spawn` for Windows) and ensure proper closing of processes.

## Phase 3: Validation & Optimization
*   [ ] Task: Verify parallel training stability (Short Run).
    *   *Goal:* Run for 100 updates with 2+ envs.
*   [ ] Task: Benchmark performance.
    *   *Goal:* Compare FPS of 1 env vs 4 envs to ensure scaling.
