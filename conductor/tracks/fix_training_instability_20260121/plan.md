# Implementation Plan - Fix Training Instability

## Strategy
We will implement centroid-based gathering guidance and switch to flat reward values. We will also adjust the training hyperparameters to favor stability over speed.

## Todo List

### 1. Update Guidance Logic
- [x] In `LevelDropBot`, modify `update_reward_states` to calculate the **centroid** of all available infantry and all available transports.
- [x] Update `previous_meeting_distances` to use distance to these centroids.
- [x] In `calculate_drop_reward`, use the centroid distance for shaping.

### 2. De-scale Rewards
- [x] Modify `calculate_drop_reward` to remove the `current_cargo_sum` multiplier from loading and unloading rewards.
- [x] Use flat `load_reward` and `unload_reward` values.

### 3. Hyperparameter Tuning
- [x] Update `config/level_drop_0.yaml`:
    - Set `learning_rate` to `0.0001`.
    - Increase `batch_size` to `128` (if possible) or ensure `n_steps` is large enough.
    - Set `target_kl` if using an algorithm that supports it, or simply lower LR.

### 4. Verification
- [x] Update `tests/test_drop_reward_logic.py` to verify that loading a unit does NOT cause a negative distance penalty for the Medivac or the group.
- [x] Verify that rewards are flat and stable.
