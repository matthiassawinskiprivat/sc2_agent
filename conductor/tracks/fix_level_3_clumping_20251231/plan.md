# Plan: Fix Level 3 Agent Clumping

## Phase 1: Fix Observation Logic
- [ ] Task: Modify `src/level_3_bot.py` to include all beacons in `enemy_units`.
- [ ] Task: Modify `src/level_3_bot.py` to set `command_target` to the optimally assigned beacon's position.
- [ ] Task: Verify the logic with a unit test (`tests/test_level_3_observation_guidance.py`).

## Phase 2: Verification
- [ ] Task: Verify visually (optional) or via detailed observation dumps that the layers are correct.
- [ ] Task: Run a short training loop (100k steps) to check for crash regressions.

## Phase 3: Finalize
- [ ] Task: Commit changes and mark track as complete.
