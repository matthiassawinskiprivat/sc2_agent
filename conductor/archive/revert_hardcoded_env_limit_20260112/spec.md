# Specification: Revert Hard-coded Env Limit and Update Configs

## Overview
The user wants to remove the hard-coded restriction in the training pipeline that forces `num_envs = 1` on Windows. Instead, they want to explicitly set `num_envs: 1` in the configuration files. This maintains stability on Windows while allowing them to easily increase the number of environments later when moving to a Linux environment.

## Functional Requirements
1.  **Remove Hard-coded Guard:**
    *   Revert the changes in `train/train_pipeline.py` that override `num_envs` to 1.
2.  **Update Configuration Files:**
    *   Locate all `config/capture_the_flag*.yaml` and `config/level_*.yaml` files.
    *   Set `num_envs: 1` in any file that currently has a value greater than 1.
3.  **Preserve Stability Fixes:**
    *   Keep the deterministic port allocation in `src/utils/single_client_manager.py` as it remains beneficial for stability.
    *   Keep the layer fallback fix in `train/envs/capture_the_flag_single_env.py`.

## Acceptance Criteria
*   **Pipeline Code Reverted:** `train/train_pipeline.py` no longer contains the `if sys.platform == "win32"` guard that forces `num_envs=1`.
*   **Configs Updated:** All relevant `.yaml` files in the `config/` directory have `num_envs` set to 1.
*   **Training Starts:** Running the pipeline results in a single environment being started (driven by config), and it doesn't crash.

## Out of Scope
*   Adding support for parallel SC2 on Windows (determined to be unstable).
