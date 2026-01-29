# Track: Verify Drop Environment Penalties

## Context
The user wants to ensure that `kill_penalty` and `damage_penalty` are correctly read from the configuration files and applied in the Drop Environment (`LevelDropEnv` or equivalent). This is crucial for training safety and preventing allied fire.

## Goals
-   Verify that `kill_penalty` is read from `config.yaml` (or specific level config).
-   Verify that `damage_penalty` is read from `config.yaml`.
-   Confirm that these values are actually used in the reward calculation logic of the Drop Environment.
-   If missing or hardcoded, fix the implementation to use the config values.

## Requirements
1.  **Configuration Reading:**
    *   The environment must look up `rewards.kill_penalty` and `rewards.damage_penalty`.
    *   It should support additive/overridden configurations (e.g., `level_drop_0.yaml` inheriting from `pipeline_drop.yaml` or `level_0.yaml`).
2.  **Reward Logic:**
    *   The reward function must deduct these penalties when an allied unit is killed or damaged by the agent (if applicable).
3.  **Tests:**
    *   Verify via inspection or a small test script that the values in the `env` object match the YAML files.

## Non-Goals
-   Changing the default values of penalties (unless they are 0/missing).
