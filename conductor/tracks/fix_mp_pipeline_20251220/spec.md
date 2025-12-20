# Track Specification: Fix and Stabilize Multi-Processing Training Pipeline

## Context
The project aims to train a StarCraft 2 reinforcement learning agent. Efficient training requires parallel environment execution (Vectorized Environments) to gather experience samples quickly. The current multi-processing implementation is broken, preventing effective training.

## Problem Statement
The multi-processing training pipeline is unstable or non-functional.
*   **Symptoms:** (To be confirmed during diagnosis) Likely crashes, hangs (deadlocks), or serialization errors when attempting to run multiple SC2 instances in parallel.
*   **Root Causes (Hypothetical):** 
    *   Windows `multiprocessing` limitations (start method defaults to 'spawn', requiring picklability).
    *   Improper handling of SC2 client processes (zombie processes, port conflicts).
    *   Shared state issues in the custom Environment class.

## Goals
1.  **Diagnose:** Identify the exact error messages and failure points in the current implementation.
2.  **Stabilize:** Refactor the environment and training pipeline to ensure robust parallel execution.
3.  **Optimize:** Ensure the solution allows for actual speedup compared to serial execution (avoiding excessive overhead).

## Scope
*   **In Scope:** 
    *   `train/train_pipeline.py`
    *   `train/envs/` (Environment classes)
    *   `train/sync_vec_env.py` / `subproc_vec_env` usage.
*   **Out of Scope:** 
    *   Changing the underlying RL algorithm (PPO) logic (unless it causes the crash).
    *   Unit micro-logic (unless it causes the crash).

## Success Criteria
*   The training script runs without crashing for at least 1000 steps using `SubprocVecEnv` (or equivalent parallel wrapper) with `n_envs > 1`.
*   SC2 clients launch and close correctly without leaving zombie processes.
*   Training metrics are successfully logged to Tensorboard during parallel execution.
