# Specification: Enable Parallel CTF Training

## Overview
Currently, the training pipeline forces `num_envs = 1` for all Capture the Flag (CTF) environments to ensure stability. With the introduction of the single-client `CaptureTheFlagSingleEnv`, scaling to multiple parallel environments is now more feasible and stable. This track aims to remove the hardcoded restriction and allow parallel training.

## Functional Requirements
1.  **Remove Pipeline Restriction:**
    *   Modify `train/train_pipeline.py` to allow `num_envs > 1` when using `CaptureTheFlagEnv` (which refers to `CaptureTheFlagSingleEnv`).
    *   Maintain the restriction for the old multi-client `CaptureTheFlagEnv` if necessary, but prioritize enabling it for the single-client version currently in use.
2.  **Verify Port Management:**
    *   Ensure that multiple parallel SC2 processes started by `SubprocVecEnv` do not collide. The library's `SC2Process` should handle this, but it must be verified.
3.  **Stability & Performance:**
    *   Ensure that `SubprocVecEnv` works correctly with `MaskablePPO` when multiple CTF environments are active.

## Acceptance Criteria
-   [ ] `train_pipeline` successfully initializes with `num_envs > 1` for CTF.
-   [ ] Multiple StarCraft II processes are observed running simultaneously during training.
-   [ ] No port conflicts or startup race conditions are observed.
-   [ ] Automated tests verify the process count matches `num_envs`.

## Out of Scope
-   Improving the internal performance of the SC2 engine itself.
-   Modifying CTF game logic.
