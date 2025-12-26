# Implementation Plan - Fix `max_steps` configuration

## Phase 1: Test & Reproduce [checkpoint: 9711853]
- [x] Task: Create reproduction test case 856a7db
    - [ ] Create `tests/test_max_steps_config.py`.
    - [ ] Implement a test that initializes `Level0Env` with a custom `max_steps` value (e.g., 100) and verifies that the environment does *not* truncate at 100 steps (reproducing the bug) or fails to accept the parameter.
- [x] Task: Conductor - User Manual Verification 'Test & Reproduce' (Protocol in workflow.md) 9711853

## Phase 2: Implementation [checkpoint: a0fcdb6]
- [x] Task: Update Level0Env signature b0927ae
    - [ ] Modify `train/envs/level_0_env.py` `__init__` method to accept `max_steps` as an argument.
    - [ ] Ensure `max_steps` has a default value (e.g., 300) for backward compatibility.
    - [ ] Update `self.max_steps` assignment to use the passed argument.
- [x] Task: Update train_pipeline.py 297712a
    - [ ] Modify `train/train_pipeline.py` to extract `max_steps` from the loaded YAML configuration (`env_params`).
    - [ ] Pass `max_steps` to the `Level0Env` constructor within the `make_env` function.
- [x] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md) a0fcdb6

## Phase 3: Verification [checkpoint: 9a29e65]
- [x] Task: Update test case d08213e
    - [ ] Update `tests/test_max_steps_config.py` to assert that `Level0Env` now respects the custom `max_steps` and truncates correctly.
    - [ ] Verify that `Level0Env` falls back to the default if no `max_steps` is provided (if applicable/tested).
- [x] Task: Run full test suite 9a29e65
    - [ ] Run `pytest` to ensure no regressions in other parts of the system.
- [x] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md) 9a29e65
