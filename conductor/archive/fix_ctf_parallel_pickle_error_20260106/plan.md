# Plan: Fix Parallel CTF Pickling Error

This plan resolves the `TypeError: cannot pickle '_thread.lock' object` crash by intercepting the masking support check in the vectorized environment wrapper.

## Phase 1: Intercept Masking Check
Update the vectorized environment to handle masking checks locally.

- [x] Task: TDD - Create a test reproducing the pickling error when calling `get_attr('action_masks')` on a `MaskableSubprocVecEnv` containing a non-picklable environment. 9dd5870
- [x] Task: Update `MaskableSubprocVecEnv.get_attr` in `train/train_pipeline.py` to intercept `'action_masks'`. 9dd5870
- [x] Task: Remove dummy `self.action_masks = None` and diagnostic warnings from `CaptureTheFlagSingleEnv` and `CaptureTheFlagEnv`. 9dd5870
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Intercept Masking Check' (Protocol in workflow.md)

## Phase 2: Final Verification
Verify that parallel training now starts and runs rollout collection correctly.

- [x] Task: Integration Test - Run `train_pipeline` with `num_envs: 2` for a short duration to confirm training begins. 9dd5870
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Final Verification' (Protocol in workflow.md)