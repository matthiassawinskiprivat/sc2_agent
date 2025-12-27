# Specification: Reward Logic Refinement and Hyperparameter Tuning

## Overview
This track aims to improve agent training performance and behavior by fixing a reward conflict (timeout vs. success), rebalancing reward magnitudes to ensure a strong success gradient over long horizons, and adding time pressure via step penalties. Additionally, hyperparameters for Level 2 will be tuned to better handle its extended episode length.

## Functional Requirements

### 1. Reward Logic Fixes (all levels)
- **Timeout vs. Success:** Modify `step()` logic to prioritize the success reward. If the agent reaches the beacon on the final step, it must receive the full success reward (+100) and NO timeout penalty. The timeout penalty (-10) should only trigger if `steps >= max_steps` AND the target has not been reached.
- **Success Reward Increase:** Change the terminal reward for reaching the beacon from +10 to **+100**.
- **Step Penalty:** Introduce a constant reward of **-0.05** per step.
- **Distance Reward Scaling:** Reduce the multiplier for the distance-based reward (difference in distance to target between steps) from 1.0 to **0.1**.

### 2. Hyperparameter Tuning (Level 2)
- **Long Horizon Optimization:** Update `config/level_2.yaml` (or the specific configuration used for Level 2) with the following SB3 PPO parameters:
    - `n_steps`: 2048 (increase batch size per environment)
    - `gamma`: 0.995 (increase discount factor to value future rewards more highly)

## Non-Functional Requirements
- **Consistency:** The reward structure should be applied consistently across `level_0_env.py`, `level_1_env.py` (if distinct), and `minimal_sc2_env.py`.
- **Maintainability:** Use configuration constants or YAML settings where possible to avoid hardcoding these values deep in the environment logic.

## Acceptance Criteria
- [ ] Reaching the beacon on the last step results in a +100 reward (verified via tests).
- [ ] Every step taken by the agent results in a -0.05 penalty being added to the reward.
- [ ] The cumulative distance reward in Level 2 no longer dwarfs the terminal success reward.
- [ ] Level 2 training configuration uses `n_steps: 2048` and `gamma: 0.995`.
- [ ] Automated tests for reward logic (`tests/test_reward_logic.py`) pass with updated values.

## Out of Scope
- Changes to the observation space or CNN architecture.
- Implementation of new environment levels.
