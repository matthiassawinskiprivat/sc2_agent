# Plan: Individual Unit Reward Refactoring

## Phase 1: Level 3/4 Foundation & Refactoring [checkpoint: b1f0b03]
- [x] Task: Create failing tests in `tests/test_level_3_reward_attribution.py` that verify intermediate rewards in the unit cycle are non-zero when progress is made.
- [x] Task: Refactor `Level3Bot.calculate_distance_reward` to accept a `unit_tag` and return the reward for that specific unit.
- [x] Task: Update `Level3Bot.check_collections` to return a mapping of `unit_tag -> reward` or allow checking for a specific unit.
- [x] Task: Modify `Level3Env._async_step` to calculate and return rewards (dist_reward + success_reward + step_penalty) for every unit in the cycle.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Level 3/4 Foundation & Refactoring' (Protocol in workflow.md)

## Phase 2: Level 7 Combat Refactoring
- [ ] Task: Create failing tests in `tests/test_level_7_reward_attribution.py` verifying that individual units in Level 7 receive rewards for their specific movements and attacks.
- [ ] Task: Refactor `Level7Bot` combat reward functions (`calculate_combat_reward`, `calculate_allied_attrition_penalty`) to attribute rewards to the active unit.
- [ ] Task: Update `Level7Bot.calculate_distance_reward` to match the individual unit pattern from Phase 1.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Level 7 Combat Refactoring' (Protocol in workflow.md)

## Phase 3: Integration & Quality Assurance
- [ ] Task: Run full test suite for Level 3, 4, and 7 to ensure no regressions in environment flow.
- [ ] Task: Verify that `previous_distances` tracking is robust when units are added or removed mid-step.
- [ ] Task: Perform a manual visual check using `run_game_with_ui.py` to observe if units commit more decisively to targets.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Quality Assurance' (Protocol in workflow.md)
