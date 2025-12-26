# Plan: Revert Reward Function to Master Branch Logic

## Phase 1: Retrieval and Analysis [checkpoint: a5fc709]
- [x] Task: Fetch `src/sc2_bot.py` from the `master` branch using `git show master:src/sc2_bot.py > src/sc2_bot_master.py`.
- [x] Task: Analyze `src/sc2_bot_master.py` to extract `_calculate_reward`, `reset_game_state` (for `previous_distance` handling), and `__init__`.
- [ ] Task: Conductor - User Manual Verification 'Retrieval and Analysis' (Protocol in workflow.md)

## Phase 2: Implementation [checkpoint: a5fc709]
- [x] Task: Update `src/sc2_bot.py` to replace `_calculate_reward` with the version from `master`.
- [x] Task: Update `src/sc2_bot.py` to ensure `previous_distance` is initialized and reset exactly as in `master`.
- [x] Task: Clean up `src/sc2_bot_master.py` temporary file.
- [ ] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md)

## Phase 3: Verification [checkpoint: a5fc709]
- [x] Task: Write a unit test `tests/test_reward_logic.py` that mocks the bot and asserts the reward output matches expected values from the "master" logic (dense reward behavior).
- [x] Task: Run the test to confirm correctness.
- [ ] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md)
