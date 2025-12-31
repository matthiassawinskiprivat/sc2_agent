# Plan: Dynamic Greedy Reward Optimization

This plan implements a globally optimal matching algorithm for reward calculation in multi-unit environments (Level 3/4).

## Phase 1: Foundation & Utilities [checkpoint: 7c455d7]
- [x] Task: Implement `scipy.optimize.linear_sum_assignment` (Hungarian Algorithm) wrapper or manual equivalent. [b1530cf]
- [x] Task: Create a utility function in `src/utils/model_utils.py` to calculate the distance matrix between two sets of points. [b1530cf]
- [x] Task: Write tests for the distance matrix and assignment logic in `tests/test_reward_logic.py`. [b1530cf]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Foundation & Utilities' (Protocol in workflow.md) [7c455d7]

## Phase 2: Bot Logic Update [checkpoint: 04a1e25]
- [x] Task: Update `Level3Bot` to store `previous_min_distances` for the optimal assignment rather than a simple per-unit nearest. [517002b]
- [x] Task: Implement `_calculate_optimal_assignment` in `Level3Bot`. [517002b]
- [x] Task: Modify `check_collections` to handle rewards based on the new assignment logic. [517002b]
- [x] Task: Write tests in `tests/test_level_3_logic.py` ensuring that if Unit A is closer to Beacon B but assigned to Beacon A, the reward reflects progress toward A. [517002b]
- [x] Task: Conductor - User Manual Verification 'Phase 2: Bot Logic Update' (Protocol in workflow.md) [04a1e25]

## Phase 3: Environment Integration [checkpoint: ebdd517]
- [x] Task: Update `Level3Env._calculate_avg_distance_reward` to use the bot's optimal assignment results. [f540016]
- [x] Task: Refactor reward configuration in `config/level_3.yaml` to support potential new weighting for the greedy reward. [f540016]
- [x] Task: Write integration tests in `tests/test_level_3_pipeline.py` to verify end-to-end reward flow. [f540016]
- [x] Task: Conductor - User Manual Verification 'Phase 3: Environment Integration' (Protocol in workflow.md) [ebdd517]

## Phase 4: Validation & Training
- [~] Task: Run a short training run (100k steps) for Level 3 and verify splitting behavior in `debug_visuals.py` or logs.
- [ ] Task: Verify that Level 4 (4 units) also benefits from the logic without additional code changes.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Validation & Training' (Protocol in workflow.md)
