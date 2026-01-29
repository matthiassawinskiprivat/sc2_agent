# Implementation Plan - Fix Hidden Drop Reward Bugs

## Strategy
We will move the "state update" logic (updating previous distances, health, and cargo) out of the reward calculation functions and into a dedicated `update_reward_states` method. This method will be called only once per full unit cycle in the environment. The reward functions themselves will only calculate differences based on these "frozen" previous states.

## Todo List

### 1. Generalize Combat Rewards
- [x] Modify `LevelDropBot.calculate_combat_reward` to track damage to **all** enemy units and structures.
- [x] Ensure `previous_health` tracking is race-agnostic and includes buildings.

### 2. Synchronize Attrition Penalties
- [x] Implement a mechanism to ensure `kill_penalty` is only applied once per unit death per cycle.
- [x] Move the deletion of dead tags from `previous_allied_health` to a post-cycle update.

### 3. Decouple Distance and Cargo Rewards
- [x] Create `update_previous_states()` in `LevelDropBot`.
- [x] Update `calculate_drop_reward` to use existing `previous_transport_distances` without updating them immediately.
- [x] Call `update_previous_states()` at the end of the unit cycle in `LevelDropEnv._async_step`.

### 4. Testing & Verification
- [x] Update `tests/test_drop_reward_logic.py` to simulate a multi-unit cycle and verify that rewards are not "stolen" or "multiplied".
- [x] Verify that destroying an enemy structure yields the expected reward.
