# Plan: Level 7 Proximity Reward and Attack Mask Optimization

## Phase 1: Reward Refactoring and Config Update
- [x] Task: Update `config/level_7.yaml` to include the `distance_multiplier: 0.05` in the rewards section. b656612
- [ ] Task: Implement `calculate_distance_reward` in `Level7Bot` using optimal assignment logic (porting from `Level3Bot`).
- [ ] Task: Integrate the proximity reward into the `Level7Env` step logic, ensuring it is summed with combat rewards.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Reward Refactoring and Config Update' (Protocol in workflow.md)

## Phase 2: Action Mask Optimization
- [ ] Task: Modify `Level7Bot.action_masks` to provide a full-True mask for the attack action segment (second half of the mask).
- [ ] Task: Verify that the Move mask (first half) remains restricted by pathing.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Action Mask Optimization' (Protocol in workflow.md)

## Phase 3: Verification and Testing
- [ ] Task: Create `tests/test_level_7_proximity_reward.py` to verify the reward calculation logic with mock unit movements.
- [ ] Task: Create `tests/test_level_7_attack_mask.py` to verify the attack mask is indeed unmasked.
- [ ] Task: Run a short debug training session using `debug_train_level_7.py` and monitor reward logs to ensure the agent approaches enemies.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification and Testing' (Protocol in workflow.md)
