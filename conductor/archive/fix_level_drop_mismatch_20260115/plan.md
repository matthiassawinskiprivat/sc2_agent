# Implementation Plan - Fix LevelDrop Model Mismatch

## Phase 1: Configuration & Environment Alignment
- [x] Task: Update `config/level_drop_0.yaml` to match project standards. [efb20aa]
    - [ ] Set `env.grid_size` to 128.
    - [ ] Set `model.policy_kwargs.net_arch` to `[]`.
- [x] Task: Refactor `LevelDropEnv.__init__` in `train/envs/level_drop_env.py`. [f04fb11]
    - [ ] Remove manual `action_heads` resolution to allow cumulative discovery from `Level0Env`.
- [x] Task: Conductor - User Manual Verification 'Configuration & Environment Alignment' (Protocol in workflow.md) [checkpoint: 9fbc35f]

## Phase 2: Verification & Testing
- [x] Task: Create a regression test `tests/test_fix_level_drop_mismatch.py`. [f644448]
    - [ ] Mock a saved model with 33 input channels and 32768 actions.
    - [ ] Verify that `train_pipeline` grafting logic can handle the 35ch/65536act transition.
- [x] Task: Run the `train_pipeline` to ensure successful initialization. [f644448]
- [x] Task: Conductor - User Manual Verification 'Verification & Testing' (Protocol in workflow.md) [checkpoint: f644448]
