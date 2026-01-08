# Plan: Fix Space Mismatches in Capture the Flag Environment

This plan focuses on restoring the "Strictly Additive" curriculum rules to the `CaptureTheFlagEnv` and `CaptureTheFlagBot`, ensuring observation and action space parity with Level 7.

## Phase 1: Environment Observation Space Restoration
Restore the missing observation layers to ensure the observation space is strictly additive.

- [ ] Task: Write failing tests for Observation Space shape in `CaptureTheFlagEnv`
- [ ] Task: Implement additive layer logic in `CaptureTheFlagEnv` to match Level 7+ requirements
- [ ] Task: Verify Observation Space coverage and parity with Level 7
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment Observation Space Restoration' (Protocol in workflow.md)

## Phase 2: Action Space Expansion
Expand the action space from a single 128x128 grid to the dual Move/Attack grid structure (32768 actions).

- [ ] Task: Write failing tests for Action Space size (32768) in `CaptureTheFlagEnv`
- [ ] Task: Update `CaptureTheFlagEnv` action space definition and `action_masks`
- [ ] Task: Verify Action Space restoration allows weight loading without grafting
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Action Space Expansion' (Protocol in workflow.md)

## Phase 3: Bot and Interaction Logic Update
Update the bot to handle the dual action head and ensure correct command dispatch.

- [ ] Task: Write failing tests for `CaptureTheFlagBot` action dispatching (Move vs Attack)
- [ ] Task: Implement dual-head action logic in `CaptureTheFlagBot.apply_action`
- [ ] Task: Refactor layer extraction in `CaptureTheFlagBot` to ensure all 9+ layers are processed correctly
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Bot and Interaction Logic Update' (Protocol in workflow.md)

## Phase 4: Full Pipeline Integration Verification
Final check to ensure the training pipeline starts the CTF level smoothly.

- [ ] Task: Execute `train_pipeline.py` and verify absence of space mismatch warnings
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Full Pipeline Integration Verification' (Protocol in workflow.md)
