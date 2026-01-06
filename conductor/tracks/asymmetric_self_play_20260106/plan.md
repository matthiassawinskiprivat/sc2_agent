# Plan: Asymmetric Self-Play for Capture the Flag

This plan implements asymmetric training in the CTF environment, focusing on Player 1 as the trainee and automating Player 2 using model snapshots.

## Phase 1: Environment & Bot Preparation
Focus on updating the `CaptureTheFlagBot` and `CaptureTheFlagEnv` to support automated inference and role identification.

- [x] Task: TDD - Add `is_trainee` flag and automated action support to `CaptureTheFlagBot` e30f16d
- [x] Task: Implement `is_trainee` logic and `set_model` integration in `CaptureTheFlagBot` e30f16d
- [ ] Task: TDD - Update `CaptureTheFlagEnv.reset` to snapshot the model for the opponent
- [ ] Task: Implement model snapshotting and opponent initialization in `CaptureTheFlagEnv.reset`
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment & Bot Preparation' (Protocol in workflow.md)

## Phase 2: Asymmetric Step Logic & Termination
Implement the core logic to automate Player 2's turns and handle the new early termination conditions.

- [ ] Task: TDD - Update `CaptureTheFlagEnv._async_step` to automate Player 2 actions
- [ ] Task: Implement automated opponent stepping in `_async_step` (skipping trainer input for P2)
- [ ] Task: TDD - Add early termination logic for trainee unit loss
- [ ] Task: Implement unit-count check and `loss_penalty` application in `_async_step`
- [ ] Task: TDD - Verify persistent objective (P2 loss does not terminate P1's turn)
- [ ] Task: Ensure environment continues for P1 after P2 units are destroyed
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Asymmetric Step Logic & Termination' (Protocol in workflow.md)

## Phase 3: Integration & Verification
Ensure the entire pipeline works together with the training configuration and perspective transformations.

- [ ] Task: TDD - Verify perspective flipping for the automated opponent
- [ ] Task: Ensure `PerspectiveTransformer` works correctly when Player 2 is automated
- [ ] Task: Integration Test - Run a short training loop to verify reward distribution and step counts
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Verification' (Protocol in workflow.md)
