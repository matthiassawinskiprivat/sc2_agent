# Track Plan: Fix Action Masking Simplex Error

## Phase 1: Diagnostics & Data Verification
*   [x] Task: Inspect `MaskableSubprocVecEnv` output. <!-- id: 0 -->
    *   *Goal:* Create a script to manually call `get_action_masks()` and print the sum, shape, and type of the result.
*   [x] Task: Add mask validation logging to `Level0Env`. <!-- id: 1 -->
    *   *Goal:* Log `sum(mask)` in `action_masks()` to see if it ever hits 0.
*   [x] Task: Conductor - User Manual Verification 'Phase 1: Diagnostics & Data Verification' (Protocol in workflow.md)

## Phase 2: Implementation & Fixes (Test-Driven)
*   [x] Task: Create `tests/test_action_mask_validity.py`. <!-- id: 2 -->
    *   *Goal:* Assert that `get_action_masks()` returns values where `np.any(mask) == True` for all envs.
*   [x] Task: Ensure `MaskableSubprocVecEnv` returns boolean dtype. <!-- id: 3 -->
    *   *Goal:* Explicitly cast to `.astype(bool)` in the wrapper to avoid integer-mask issues in SB3.
*   [x] Task: Implement Fallback in `Level0Env.action_masks()`. <!-- id: 4 -->
    *   *Goal:* If all actions are False, return an all-True mask (or a single True action at center) to prevent crash.
*   [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation & Fixes (Test-Driven)' (Protocol in workflow.md)

## Phase 3: Validation
*   [x] Task: Verify fix with short training run. <!-- id: 5 -->
    *   *Goal:* Run `train_pipeline.py` for 2048 steps with 2 environments.
*   [x] Task: Verify mask quality. <!-- id: 6 -->
    *   *Goal:* Ensure the masks are actually masking invalid terrain and not just defaulting to all-True.
*   [x] Task: Conductor - User Manual Verification 'Phase 3: Validation' (Protocol in workflow.md)