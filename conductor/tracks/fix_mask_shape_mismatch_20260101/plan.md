# Implementation Plan: Fix Action Mask Shape Mismatch in Level 7

This plan addresses the `RuntimeError: shape '[4, 32768]' is invalid for input of size 65536` by ensuring the action mask generation logic matches the expanded action space in Level 7.

## Phase 1: Test Case Reproduction [checkpoint: fdb6839]
- [x] Task: Create reproduction test `tests/test_level_7_mask_mismatch.py`. fdb6839
    - [x] Sub-task: Instantiate `Level7Env`.
    - [x] Sub-task: Call `step()` or `_async_step()` with a mock action.
    - [x] Sub-task: Assert that the action mask shape matches `env.action_space.n`.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Test Case Reproduction' (Protocol in workflow.md)

## Phase 2: Core Logic Implementation [checkpoint: 0893034]
- [x] Task: Update mask generation in `src/level_7_bot.py`. 0893034
    - [x] Sub-task: Ensure `get_observation_for_unit` (or the relevant mask-providing method) returns a mask of size `2 * GridSize^2`.
    - [x] Sub-task: Populate the first half with valid Move actions.
    - [x] Sub-task: Populate the second half with valid Attack actions (can be identical to Move or fully open for now).
- [x] Task: Enhance environment validation in `train/envs/level_7_env.py`. 0893034
    - [x] Sub-task: Add a check in `_async_step` (or where the mask is processed) to verify the mask shape matches `self.action_space.n`.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Core Logic Implementation' (Protocol in workflow.md)

## Phase 3: Final Verification [checkpoint: d5d50c8]
- [x] Task: Run all Level 7 integration tests. d5d50c8
    - [x] Sub-task: `pytest tests/test_level_7_env.py`
    - [x] Sub-task: `pytest tests/test_level_7_mask_mismatch.py`
- [x] Task: Run a short local training session to verify SB3 compatibility. d5d50c8
    - [x] Sub-task: Execute `python debug_train.py --level level_7 --steps 100`. (Verified with `debug_train_level_7.py`)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Final Verification' (Protocol in workflow.md)
