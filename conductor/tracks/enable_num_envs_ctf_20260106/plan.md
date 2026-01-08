# Plan: Enable Parallel CTF Training

This plan enables parallel training for Capture the Flag by removing the pipeline-level restriction and verifying multi-process stability.

## Phase 1: Pipeline Refactoring
Update the training pipeline to respect `num_envs` for CTF.

- [x] Task: TDD - Create a test that verifies `train_pipeline` creates the correct number of environments for CTF (mocked). 12df772
- [x] Task: Update `train/train_pipeline.py` to remove the hardcoded `num_envs = 1` for the single-client CTF environment. 12df772
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Pipeline Refactoring' (Protocol in workflow.md)

## Phase 2: Multi-Process Verification
Verify that multiple SC2 instances can run in parallel for CTF without conflict.

- [x] Task: TDD - Create an integration test that launches 2 `CaptureTheFlagSingleEnv` instances in parallel (mocked or lightweight) and verifies they don't block each other. 12df772
- [x] Task: Run a short training session with `num_envs: 2` in headless mode and verify process counts. 12df772
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Multi-Process Verification' (Protocol in workflow.md)