# Implementation Plan - Optimize Drop Reward Logic

## Strategy
We will modify `LevelDropBot` and `Level7Bot` to implement the requested reward refinements. The key is to override or adjust the combat and attrition logic inherited from `Level7Bot` to be aware of the "Drop" context (passengers and transport-specific rewards).

## Todo List

### 1. Preparation & Testing
- [x] Create a regression test `tests/test_drop_reward_logic.py` that simulates loading a unit and moving a transport to verify current (broken) reward behavior.

### 2. Fix "Fake Death" Penalty
- [x] Modify `Level7Bot.calculate_allied_attrition_penalty` or override it in `LevelDropBot` to include passengers in the "living" unit set.
- [x] Ensure that when a unit transitions from map to cargo, it isn't counted as a death.

### 3. Implement Weighted Distance Reward
- [x] Refine `LevelDropBot.calculate_drop_reward` to handle the weighted distance logic for transports.
- [x] Ensure `Level7Bot.calculate_distance_reward` is either disabled or correctly bypassed for the drop environment to avoid double-counting or conflicting rewards.

### 4. Exclude Infantry from Distance Reward
- [x] In `LevelDropBot`, override `calculate_distance_reward` (or the logic that calls it) to ensure only transports (and only when active) are considered for distance shaping.

### 5. Verification
- [x] Run `tests/test_drop_reward_logic.py` to confirm the fixes.
- [x] Run existing integration tests `tests/integration_test_level_drop_init.py` to ensure no regressions.
- [x] Verify multi-race compatibility (Terran/Zerg/Protoss types).

## Success Criteria
- Loading a unit yields `load_reward` and NO `kill_penalty`.
- Empty transports get neutral or low distance reward.
- Loaded transports get high distance reward for moving toward the objective.
- Infantry units do not get rewards for moving toward the objective (they should wait for the transport).
