# Plan: Fix train_pipeline TypeError and Improve IPC Error Handling

## Phase 1: Fix Signature and Basic Robustness
- [x] Task: Create unit test to verify `SC2Bot.reset_game_state` accepts the `verify` parameter.
- [x] Task: Update `SC2Bot.reset_game_state` in `src/sc2_bot.py` to include `verify: bool = True`.
- [x] Task: Add defensive checks (null/empty checks) in `Level3Bot.get_closest_targets` and `Level7Bot.get_closest_targets_combat` to prevent crashes on empty collections.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Fix Signature and Basic Robustness' (Protocol in workflow.md) [checkpoint: Phase 1 Done]

## Phase 2: Improved IPC Logging
- [x] Task: Enhance `Level0Env.reset` in `train/envs/level_0_env.py` to provide better context logs when an error occurs during the setup session.
- [x] Task: Update `train/train_pipeline.py` to catch `EOFError`/`BrokenPipeError` during `env.reset()` and output a more informative message directing the user to worker logs.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Improved IPC Logging' (Protocol in workflow.md) [checkpoint: Phase 2 Done]

## Phase 3: Integration and Verification
- [x] Task: Run greedy logic tests (`tests/test_level_3_greedy_logic.py`, `tests/test_level_7_greedy_logic.py`) with empty target scenarios.
- [x] Task: Verify the full `train_pipeline.py` startup flow.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Integration and Verification' (Protocol in workflow.md) [checkpoint: Phase 3 Done]
