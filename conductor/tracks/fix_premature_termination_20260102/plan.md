# Plan: Fix Premature Episode Termination

## Phase 1: Logic Verification & Failing Tests
- [x] Task: Create a reproduction test in `tests/test_termination_repro.py` that mocks unit behavior and verifies if `Level3Env._async_step` or `Level7Env._async_step` returns `done=True` prematurely.
- [x] Task: Verify if the `WSMessageTypeError` is being caught by `Level0Env.step` and triggering a reset by adding debug logging to the exception handler.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Logic Verification & Failing Tests' (Protocol in workflow.md)

## Phase 2: Environment & Bot Fixes
- [x] Task: Refactor `Level3Env._async_step` to ensure `done` logic is robust against mid-cycle unit removal and that the cycle correctly restarts or advances without termination unless terminal conditions are met.
- [x] Task: Refactor `Level7Env._async_step` to align with the fixed logic from Level 3.
- [x] Task: Update `Level3Bot` and `Level7Bot` to ensure `get_active_zerglings` / `get_active_units` correctly handles units that were just killed in the same step.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Environment & Bot Fixes' (Protocol in workflow.md)

## Phase 3: Integration & Stability Check
- [x] Task: Execute a dry-run training script (e.g., `debug_train.py`) and monitor worker logs to ensure episodes last for multiple steps.
- [x] Task: Verify that the `WSMessageTypeError` no longer occurs once resets are infrequent.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Integration & Stability Check' (Protocol in workflow.md)
