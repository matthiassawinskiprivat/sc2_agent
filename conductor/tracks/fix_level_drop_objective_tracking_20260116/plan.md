# Plan: Fix Level Drop Objective Tracking

This plan addresses the issue where `LevelDropBot` fails to find the enemy main structure at initialization, breaking the command objective layer and reward logic.

## Phase 1: Verification & Failing Test (Red Phase)
- [ ] Task: Create a reproduction test `tests/test_level_drop_objective_recovery.py` that mocks a scenario where the enemy structure is missing at step 0 but appears at step 5.
- [ ] Task: Confirm the test fails as `LevelDropBot` currently only checks for the objective during `reset_game_state`.

## Phase 2: Implementation (Green Phase)
- [ ] Task: Refactor `LevelDropBot.reset_game_state` to initialize objective tracking variables without the warning if the structure is missing.
- [ ] Task: Implement a lazy-loading/continuous scan mechanism in `LevelDropBot` (e.g., in `get_observation_for_unit`) that attempts to find the objective if `self._objective_pos` is None.
- [ ] Task: Update the logging to be less spammy while still informing when the objective is eventually found.
- [ ] Task: Run the reproduction test and confirm it passes.

## Phase 3: Integration & Validation
- [ ] Task: Verify that `raw_obs['command_target']` is correctly propagated to the `command_objective` layer using the existing `test_unified_observation.py` logic as a reference.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)
