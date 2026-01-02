# Plan: Greedy Closest-Target Logic for Level 3 and Level 7

## Phase 1: Level 3 Environment Refactor (Navigation)
- [x] Task: Create failing tests for Level 3 greedy logic. Verify that reward calculation and observation target now prioritize the closest target regardless of other units.
- [x] Task: Implement `get_closest_targets` in `Level3Bot` to replace the assignment-based `_calculate_optimal_assignment`.
- [x] Task: Update `get_observation_for_unit` in `Level3Bot` to provide the closest target's position to the agent system.
- [x] Task: Update `calculate_distance_reward` in `Level3Bot` to use the average distance change to closest targets.
- [ ] Task: Conductor - User Manual Verification 'Level 3 Environment Refactor' (Protocol in workflow.md)

## Phase 2: Level 7 Environment Refactor (Combat)
- [x] Task: Create failing tests for Level 7 greedy logic. Focus on multiple allied units converging on the same closest enemy.
- [x] Task: Implement `get_closest_targets_combat` in `Level7Bot` to replace `_calculate_optimal_assignment_combat`.
- [x] Task: Update `get_observation_for_unit` in `Level7Bot` to use the closest enemy as the command target.
- [x] Task: Update `calculate_distance_reward` in `Level7Bot` to reflect average progress towards closest enemies.
- [ ] Task: Conductor - User Manual Verification 'Level 7 Environment Refactor' (Protocol in workflow.md)

## Phase 3: Cleanup & Final Validation
- [ ] Task: Run full suite of Level 3 and Level 7 integration tests to ensure no regressions in training flow.
- [ ] Task: Clean up deprecated assignment-related methods in `Level3Bot` and `Level7Bot`.
- [ ] Task: Conductor - User Manual Verification 'Cleanup & Final Validation' (Protocol in workflow.md)
