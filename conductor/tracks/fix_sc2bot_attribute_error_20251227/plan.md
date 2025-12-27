# Plan: Fix SC2Bot AttributeError (enemy_race)

## Phase 1: Fix and Unit Verification [checkpoint: f985b0f]
Restore the missing attribute and verify the fix at the unit level.

- [x] Task: Update `tests/test_reward_logic.py` (or create a specific instantiation test) to verify `SC2Bot` has the `enemy_race` attribute. f985b0f
- [x] Task: Implement/Verify the `enemy_race` initialization fix in `src/sc2_bot.py`. f985b0f
- [x] Task: Run unit tests to ensure `SC2Bot` can be instantiated and basic logic works. f985b0f
- [x] Task: Conductor - User Manual Verification 'Fix and Unit Verification' (Protocol in workflow.md) f985b0f

## Phase 2: Pipeline Integration Verification [checkpoint: f985b0f]
Verify that the fix resolves the crash in the full training pipeline across different configurations.

- [x] Task: Run `train_pipeline.py` with `level_0.yaml` for 1000 steps to verify initialization success. f985b0f
- [x] Task: Run `train_pipeline.py` with `level_1.yaml` for a short run to verify stability. f985b0f
- [x] Task: Run `train_pipeline.py` with `level_2.yaml` for a short run to verify stability. f985b0f
- [x] Task: Conductor - User Manual Verification 'Pipeline Integration Verification' (Protocol in workflow.md) f985b0f