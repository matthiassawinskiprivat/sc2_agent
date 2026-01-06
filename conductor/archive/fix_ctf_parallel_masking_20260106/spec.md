# Specification: Fix Parallel CTF Masking Support

## Overview
Enabling `num_envs > 1` for Capture the Flag (CTF) causes a `BrokenPipeError`/`EOFError` in the training pipeline. This happens because `MaskablePPO` checks for masking support by calling `env.get_attr('action_masks')`. In a `SubprocVecEnv`, this results in an attempt to pickle a bound method across process boundaries, causing the worker process to crash silently.

## Functional Requirements
1.  **Avoid Method Pickling:**
    *   Ensure that `env.get_attr('action_masks')` does not result in a crash when using `MaskableSubprocVecEnv`.
    *   This can be achieved by making `action_masks` an attribute (e.g., a dummy value) that satisfies SB3's check, while keeping the actual logic in the `action_masks()` method.
2.  **Robust Masking Detection:**
    *   Ensure `is_masking_supported(env)` returns True for the vectorized environment without causing remote process crashes.
3.  **Clean Diagnostic Logic:**
    *   Remove the "SHADOWED" critical warning from `reset()` as it is expected behavior for `BotAI` instances and might clutter logs or cause confusion.

## Acceptance Criteria
-   [ ] `train_pipeline` starts successfully with `num_envs: 2` for CTF.
-   [ ] `is_masking_supported(env)` returns True for the vectorized CTF environment.
-   [ ] No `BrokenPipeError` or `EOFError` occurs during the first rollout collection.

## Out of Scope
-   Changing the internal implementation of `SubprocVecEnv` in Stable Baselines 3.
