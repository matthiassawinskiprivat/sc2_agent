# Plan: Fix Level 3 Agent Clumping

## Phase 1: Fix Observation Logic
- [x] Task: Modify `src/level_3_bot.py` to include all beacons in `enemy_units`. [3fe924a]
- [x] Task: Modify `src/level_3_bot.py` to set `command_target` to the optimally assigned beacon's position. [3fe924a]
- [x] Task: Verify the logic with a unit test (`tests/test_level_3_observation_guidance.py`). [3fe924a]

## Phase 2: Verification
- [ ] Task: Verify visually (optional) or via detailed observation dumps that the layers are correct.
- [ ] Task: Run a short training loop (100k steps) to check for crash regressions.

## Phase 3: Finalize
- [ ] Task: Commit changes and mark track as complete.
