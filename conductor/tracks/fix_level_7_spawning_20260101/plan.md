# Plan: Fix Level 7/8 Spawning Reliability and Unit Ratios

## Phase 1: Logic Implementation
- [x] Task: Update `Level7Bot.reset_game_state` to implement the 2-to-1 Zergling spawning ratio. 164fa22
- [x] Task: Add `await self.client.step()` after unit cleanup in `Level7Bot.reset_game_state` to ensure reliable spawning. 164fa22
- [x] Task: Ensure `debug_show_map()` is called before unit cleanup to allow the bot to see all units to kill. feb145f
- [~] Task: Conductor - User Manual Verification 'Phase 1: Logic Implementation' (Protocol in workflow.md)

## Phase 2: Verification and Testing
- [ ] Task: Create `tests/test_level_7_spawning_reliability.py` to verify correct unit counts for various configurations (Zergling vs non-Zergling).
- [ ] Task: Run `debug_train_level_7.py` with `targets: 2` and confirm 2 Roaches and 4 Zerglings spawn consistently.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Verification and Testing' (Protocol in workflow.md)
