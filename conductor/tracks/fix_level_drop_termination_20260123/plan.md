# Plan: Fix Level Drop Termination Logic

**Objective:** Ensure the game ends immediately when the enemy Command Center (or main structure) is destroyed in the Level Drop environment, and the agent receives the win reward.

**Context:** 
The user reported that the game continues even after the agent destroys the enemy Command Center. A previous attempt to add a check for `enemy_townhalls.amount == 0` in `LevelDropEnv` was made, but the user indicated it didn't work (or they saw it fail). The goal is to ensure this termination condition is robust and actually triggers in the live environment.

**Status:** Completed

## 1. Investigation
- [x] Review `train/envs/level_drop_env.py` to confirm the current logic is deployed.
- [x] Analyze why `enemy_structures.filter(...)` might fail to detect destruction.
    - Added lifted structure variants (FLYING).
- [x] Check if `LevelDropBot` state update (`_refresh_state` or similar) is properly syncing `enemy_structures` to the Env's `_bot`.

## 2. Reproduction
- [x] Create a test that simulates a game state where the CC is dead but other units exist, and verify `_async_step` returns `done=True`.
- [x] Added visible (green) logging for live verification by the user.

## 3. Implementation
- [x] Refine the termination condition in `LevelDropEnv`.
    - Added `COMMANDCENTERFLYING`, `ORBITALCOMMANDFLYING`, etc. to townhall types.
    - Implemented explicit townhall count check.

## 4. Verification
- [x] Run existing tests: `tests/test_level_drop_termination.py`.
- [x] Manual verification by user confirmed: "I saw the message and the game terminated".
