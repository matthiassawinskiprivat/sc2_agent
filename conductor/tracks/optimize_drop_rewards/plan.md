# Plan: Optimize Drop Rewards

- [x] **Code Modification**
    - [x] Update `src/level_drop_bot.py`:
        -   Implement `progressive_load_reward` logic.
        -   Implement `cargo_weighted_distance_reward` logic.
        -   Ensure `previous_distances` are tracked correctly for Medivacs.

- [x] **Configuration**
    - [x] Update `config/level_drop_0.yaml` to include new reward parameters if we decide to expose them, or just rely on the existing `load_reward` as a base.

- [x] **Verification**
    - [x] Create a unit test `tests/test_drop_rewards.py` to simulate a scenario and verify the calculated reward increases as cargo increases and distance decreases.
