# Specification - Fix Level 0 Environment Pickle Error

## Overview
The training pipeline fails on Windows due to non-picklable lambda functions used in environment creation within `train_pipeline.py`.

## Issues
1. `ActionMasker(env, lambda e: e.unwrapped.action_masks())` uses a lambda that cannot be pickled by `multiprocessing`.
2. `make_env` returns a local function `_init`, which is also problematic on Windows.

## Functional Requirements
* **Picklable Masker:** Replace the lambda with a static method or a top-level function.
* **Picklable Env Factory:** Replace local `_init` with a picklable class or top-level function.
* **Compatibility:** Ensure all environment types (Level 0, 3, 7, CTF) are supported.

## Acceptance Criteria
* `train_pipeline.py` launches successfully for `level_0` with `num_envs > 1`.
* No `AttributeError: Can't pickle local object` occurs during worker initialization.
