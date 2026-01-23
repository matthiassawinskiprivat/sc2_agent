# Implementation Plan - Fix Reward Explosion & Attribution

## Strategy
We will transition from "global state differences" to "per-unit state differences" for load/unload rewards. We will maintain `previous_allied_tags` and `previous_transport_cargo` (per tag) to detect exactly when a specific unit changes its relationship with a transport. We will also normalize the configuration values.

## Todo List

### 1. Update Configuration
- [x] Modify `config/level_drop_0.yaml` to reduce `damage_reward` (e.g., to `0.05`).
- [x] Adjust `kill_reward` and `win_reward` if necessary for better balance.

### 2. Implement Per-Unit Tracking
- [x] Add `self.previous_transport_cargo = {}` and `self.previous_allied_tags = set()` to `LevelDropBot`.
- [x] Update these in `update_reward_states`.

### 3. Refine Attribution Logic
- [x] Modify `LevelDropBot.calculate_drop_reward` to:
    - Reward a Marine if it was on the map and is now a passenger.
    - Reward a Medivac if its `cargo_used` increased compared to its *own* previous state.
    - Only reward "Arrival" if the Medivac is the active unit and just performed an unload.

### 4. Verification
- [x] Update `tests/test_drop_reward_logic.py` to verify that loading 8 units results in exactly 16 rewards (8 for Marines, 8 for Medivac) rather than hundreds.
- [x] Verify that structure damage rewards are now within a sensible range.
