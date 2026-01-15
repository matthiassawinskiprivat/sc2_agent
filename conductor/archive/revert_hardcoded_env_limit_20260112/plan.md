# Plan: Revert Hard-coded Env Limit and Update Configs

## Phase 1: Revert Code Changes
- [x] Task: Remove the `sys.platform == "win32"` guard in `train/train_pipeline.py` that forces `num_envs=1`. 72b5093
- [x] Task: Conductor - User Manual Verification 'Revert Code Changes' (Protocol in workflow.md) 72b5093

## Phase 2: Update Configurations
- [x] Task: Identify and update all `config/*.yaml` files to set `num_envs: 1`. 934edf2
- [x] Task: Conductor - User Manual Verification 'Update Configurations' (Protocol in workflow.md) 934edf2

## Phase 3: Final Verification
- [x] Task: Run a short training test to ensure it respects the config and remains stable with `num_envs: 1`. 934edf2
- [x] Task: Conductor - User Manual Verification 'Final Verification' (Protocol in workflow.md) 934edf2