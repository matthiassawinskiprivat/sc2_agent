# Plan: Fix Beacon Visibility in Observation

## Phase 1: Investigation and Reproduction [x] 90535f9
- [x] Task: Research configuration loading in `train_pipeline.py`. Verify how `level_0.yaml` is merged and passed to environments and transformers.
- [x] Task: Audit `src/transformers/observation_transformer.py` to identify how the `enemy_units` layer is populated and if special handling for Beacons is present/missing.
- [x] Task: Write TDD Test - Create `tests/test_beacon_visibility_regression.py`. This test should mock a StarCraft 2 observation containing a Beacon and assert that the `observation_transformer` places it on the `enemy_units` layer.
- [x] Task: Implement/Update Reproduction Script - Ensure `tests/test_beacon_visibility_regression.py` fails (Red Phase).
- [x] Task: Conductor - User Manual Verification 'Investigation and Reproduction' (Protocol in workflow.md)

## Phase 2: Fix Implementation [checkpoint: 90535f9]
- [x] Task: Fix `src/sc2_bot.py` to ensure Beacon and Zergling are always respawned during `reset_game_state`, even if they are missing/dead.
- [x] Task: Conductor - User Manual Verification 'Fix Implementation' (Protocol in workflow.md)

## Phase 3: Integration and Regression Testing [x]
- [x] Task: Run `train_pipeline.py` with a short duration/headless mode to ensure no initialization crashes occur with the new fixes. (Confirmed by user)
- [x] Task: Verify that all existing unit tests pass to ensure no regressions in other observation layers.
- [x] Task: Conductor - User Manual Verification 'Integration and Regression Testing' (Protocol in workflow.md)
