# Specification: Fix Parallel CTF Pickling Error

## Overview
Enabling `num_envs > 1` for Capture the Flag (CTF) causes a `TypeError: cannot pickle '_thread.lock' object` in the training pipeline. This happens because SB3's `MaskablePPO` checks for masking support by calling `env.get_attr('action_masks')`. In a `SubprocVecEnv`, this results in an attempt to fetch and pickle a bound method from the remote process. Since the environment contains non-picklable objects (like an `asyncio` event loop with locks), pickling the bound method fails as it tries to pickle its associated environment instance.

## Functional Requirements
1.  **Intercept Masking Check:**
    *   Override `get_attr` in `MaskableSubprocVecEnv` (within `train/train_pipeline.py`) to intercept requests for `'action_masks'`.
    *   Return a simple, picklable dummy value (e.g., `True`) directly from the main process, satisfying SB3's support check without communicating with the workers.
2.  **Clean Up Environment classes:**
    *   Remove any dummy `action_masks` attributes or diagnostic "SHADOWED" warnings from `CaptureTheFlagSingleEnv` and `CaptureTheFlagEnv`, as they are redundant or noisy with the new interception logic.
3.  **Maintain Functional Masking:**
    *   Ensure that `get_action_masks()` (which uses `env_method("action_masks")`) continues to function correctly by calling the actual method on the remote workers and returning the picklable numpy result.

## Acceptance Criteria
-   [ ] `train_pipeline` starts successfully with `num_envs: 2` for CTF.
-   [ ] `is_masking_supported(env)` returns True without triggering a cross-process pickling error.
-   [ ] Rollout collection begins and training proceeds without `BrokenPipeError` or `EOFError`.

## Out of Scope
-   Modifying Stable Baselines 3 or Gymnasium core libraries.
