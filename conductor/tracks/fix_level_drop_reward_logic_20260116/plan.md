# Implementation Plan - Fix Level Drop Reward Logic

## Phase 1: Test Infrastructure & Reproduction [checkpoint: 769d0a2]
- [x] Task: Create reproduction test suite `tests/test_level_drop_reward_logic.py`. [89f0f7a]
    - [ ] Mock `LevelDropBot` state with allied units near a known enemy position.
    - [ ] Simulate Fog of War (position `_objective_pos` known but unit not in `enemy_structures`).
    - [ ] Assert that `calculate_drop_reward` currently returns 0.0 (reproducing the bug).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Test Infrastructure & Reproduction' (Protocol in workflow.md)

## Phase 2: Implementation
- [x] Task: Update `LevelDropBot.calculate_drop_reward` logic. [1ea8693]
    - [x] Modify distance calculation to prioritize `self._objective_pos`.
    - [x] Implement fallback to `self.enemy_structures.find_by_tag` if `_objective_pos` is missing.
    - [x] Ensure units are correctly filtered (not flying and within 15.0 radius).
- [x] Task: Verify fix with reproduction tests. [1ea8693]
    - [x] Run `pytest tests/test_level_drop_reward_logic.py`.
    - [x] Confirm all tests pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Integration & Quality Gate
- [ ] Task: Integration Verification.
    - [ ] Run existing tests: `tests/test_level_drop_env.py` and `tests/test_level_drop_system.py`.
    - [ ] Verify no regressions in reward accumulation.
- [ ] Task: Final Quality Gate.
    - [ ] Check code coverage for `src/level_drop_bot.py`.
    - [ ] Ensure adherence to project code style.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Quality Gate' (Protocol in workflow.md)
