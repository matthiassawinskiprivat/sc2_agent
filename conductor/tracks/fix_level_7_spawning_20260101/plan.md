# Plan: Fix Level 7/8 Spawning Reliability and Unit Ratios

## Phase 1: Robust Base Spawning [checkpoint: 626b23e]
- [x] Task: Update `Level7Bot.reset_game_state` to implement the 2-to-1 Zergling spawning ratio. 164fa22
- [x] Task: Add `await self.client.step()` after unit cleanup in `Level7Bot.reset_game_state` to ensure reliable spawning. 164fa22
- [x] Task: Ensure `debug_show_map()` is called before unit cleanup to allow the bot to see all units to kill. feb145f
- [x] Task: Implement a robust cleanup loop in `reset_game_state` that verifies all units are gone before spawning. fbe7a45
- [x] Task: Implement post-spawn verification and retry logic to ensure configured unit counts are strictly met. fbe7a45
- [x] Task: Conductor - User Manual Verification 'Phase 1: Robust Base Spawning' (Protocol in workflow.md) 626b23e

## Phase 2: Verification and Cleanup
- [~] Task: Update unit tests in `tests/test_level_7_spawning_reliability.py` to verify centralized logic and unit counts.
- [ ] Task: Verify that the 2-to-1 Zergling ratio is correctly handled in the new logic.
- [ ] Task: Run a short debug training session and monitor for "Spawning attempt... failed" logs to ensure loops are working.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Verification and Cleanup' (Protocol in workflow.md)