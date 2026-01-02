# Plan: Fix Agent Immobility in Level 3 and 7

## Phase 1: Root Cause Analysis & Reproduction
- [x] Task: Create a reproduction test `tests/test_unit_movement_regression.py` that specifically asserts that unit positions change after an action is applied.
- [x] Task: Add temporary debug logging in `Level3Bot` and `Level7Bot` to trace the action flow from `apply_action` to the `SC2` command queue.
- [x] Task: Conductor - User Manual Verification 'Root Cause Analysis & Reproduction' (Protocol in workflow.md)

## Phase 2: Fix Action Flow
- [x] Task: Refactor the action transmission logic in `Level3Env` and `Level7Env` (or the bots) to ensure commands are correctly queued and stepped through the `SC2` client.
- [~] Task: Verify the fix using the `tests/test_unit_movement_regression.py` (ensure it transitions from Red to Green).
- [ ] Task: Conductor - User Manual Verification 'Fix Action Flow' (Protocol in workflow.md)

## Phase 3: Final Verification
- [x] Task: Run `debug_train_level_3.py` and `debug_train_level_7.py` to visually (or via logs) confirm unit movement and reward accumulation.
- [x] Task: Execute all existing integration tests (`tests/test_level_3_integration.py`, `tests/test_level_7_integration.py`) to ensure overall stability.
- [x] Task: Conductor - User Manual Verification 'Final Verification' (Protocol in workflow.md)
