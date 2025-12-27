# Plan: Reward Logic Refinement and Hyperparameter Tuning

## Phase 1: Environment Logic & Reward Refinement
Refactor the reward calculation logic across all environments to fix the timeout bug, increase success rewards, and add step penalties.

- [ ] Task: Update `tests/test_reward_logic.py` to reflect the new reward magnitudes (Success: +100, Step: -0.05, Distance Multiplier: 0.1).
- [ ] Task: Implement the "Timeout Penalty Overwrite" fix in `src/train/envs/level_0_env.py` and `src/train/envs/minimal_sc2_env.py`.
- [ ] Task: Apply the updated reward magnitudes and step penalty to `src/train/envs/level_0_env.py`.
- [ ] Task: Apply the updated reward magnitudes and step penalty to `src/train/envs/minimal_sc2_env.py`.
- [ ] Task: Verify reward logic with updated tests.
- [ ] Task: Conductor - User Manual Verification 'Environment Logic & Reward Refinement' (Protocol in workflow.md)

## Phase 2: Configuration & Hyperparameter Tuning
Update the configuration files to reflect the new rewards and tune Level 2 for long-horizon learning.

- [ ] Task: Update `config/level_0.yaml`, `config/level_1.yaml`, and `config/level_2.yaml` with the new reward constants if they are exposed there.
- [ ] Task: Update `config/level_2.yaml` with `n_steps: 2048` and `gamma: 0.995`.
- [ ] Task: Verify that the `ConfigManager` correctly loads the new Level 2 hyperparameters.
- [ ] Task: Conductor - User Manual Verification 'Configuration & Hyperparameter Tuning' (Protocol in workflow.md)

## Phase 3: Integration & Performance Validation
Ensure the changes work correctly in a full training loop and don't introduce regressions.

- [ ] Task: Run a short integration test for Level 0 to verify the reward accumulation in the logs.
- [ ] Task: Run a short integration test for Level 2 to verify that `n_steps` and `gamma` are correctly passed to the SB3 PPO model.
- [ ] Task: Verify Tensorboard logs show the expected reward magnitudes during these short runs.
- [ ] Task: Conductor - User Manual Verification 'Integration & Performance Validation' (Protocol in workflow.md)
