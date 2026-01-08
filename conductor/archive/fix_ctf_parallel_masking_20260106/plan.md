# Plan: Fix Parallel CTF Masking Support

This plan resolves the communication crash between the main process and workers when training CTF in parallel by fixing how action masking support is detected.

## Phase 1: Environment Fixes
Update the environment to satisfy masking checks without crashing.

- [x] Task: TDD - Create a test that reproduces the crash when calling `get_attr('action_masks')` on a `SubprocVecEnv` containing `CaptureTheFlagSingleEnv`. fd64bc6
- [x] Task: Add a dummy `action_masks` attribute to `CaptureTheFlagSingleEnv` to satisfy `get_attr` checks. fd64bc6
- [x] Task: Remove the diagnostic "SHADOWED" warning from `reset()` methods in CTF environments. fd64bc6
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment Fixes' (Protocol in workflow.md)

## Phase 2: Integration Verification
Verify that parallel training now starts and runs rollout collection correctly.

- [x] Task: Integration Test - Run `train_pipeline` with `num_envs: 2` for a very short duration (mocked or headless) to verify rollout collection begins. fd64bc6
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Integration Verification' (Protocol in workflow.md)
