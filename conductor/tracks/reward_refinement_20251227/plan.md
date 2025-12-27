# Plan: Reward Logic Refinement and Hyperparameter Tuning

## Phase 1: Environment Logic & Reward Refinement [checkpoint: e3be2a4]
Refactor the reward calculation logic across all environments to fix the timeout bug, increase success rewards, and add step penalties.

- [x] Task: Update `tests/test_reward_logic.py` to reflect the new reward magnitudes (Success: +100, Step: -0.05, Distance Multiplier: 0.1). e3be2a4
- [x] Task: Implement the "Timeout Penalty Overwrite" fix in `src/train/envs/level_0_env.py` and `src/train/envs/minimal_sc2_env.py`. e3be2a4
- [x] Task: Apply the updated reward magnitudes and step penalty to `src/train/envs/level_0_env.py`. e3be2a4
- [x] Task: Apply the updated reward magnitudes and step penalty to `src/train/envs/minimal_sc2_env.py`. e3be2a4
- [x] Task: Verify reward logic with updated tests. e3be2a4
- [x] Task: Conductor - User Manual Verification 'Environment Logic & Reward Refinement' (Protocol in workflow.md) e3be2a4

## Phase 2: Configuration & Hyperparameter Tuning [checkpoint: e3be2a4]
Update the configuration files to reflect the new rewards and tune Level 2 for long-horizon learning.     

- [x] Task: Update `config/level_0.yaml`, `config/level_1.yaml`, and `config/level_2.yaml` with the new reward constants if they are exposed there. e3be2a4
- [x] Task: Update `config/level_2.yaml` with `n_steps: 2048` and `gamma: 0.995`. e3be2a4
- [x] Task: Verify that the `ConfigManager` correctly loads the new Level 2 hyperparameters. e3be2a4
- [x] Task: Conductor - User Manual Verification 'Configuration & Hyperparameter Tuning' (Protocol in workflow.md) e3be2a4

## Phase 3: Integration & Performance Validation [checkpoint: e3be2a4]
Ensure the changes work correctly in a full training loop and don't introduce regressions.

- [x] Task: Run a short integration test for Level 0 to verify the reward accumulation in the logs. e3be2a4
- [x] Task: Run a short integration test for Level 2 to verify that `n_steps` and `gamma` are correctly passed to the SB3 PPO model. e3be2a4
- [x] Task: Verify Tensorboard logs show the expected reward magnitudes during these short runs. e3be2a4
- [x] Task: Conductor - User Manual Verification 'Integration & Performance Validation' (Protocol in workflow.md) e3be2a4