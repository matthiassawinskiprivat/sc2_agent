# Plan: Graceful Termination Handling ("Game has already ended")

This plan outlines the steps to prevent crashes during game end in the Capture the Flag environment and ensure correct final reward attribution.

## Phase 1: Robust Error Detection and Bot Safeguards
Implement catching of `ProtocolError` and ensure bots don't crash when game objects disappear.

- [x] Task: Write failing tests for `ProtocolError` handling in `MultiClientEnv`
- [x] Task: Update `MultiClientEnv.step` to catch "Game has already ended" and flag termination
- [x] Task: Add defensive checks to `CaptureTheFlagBot` for missing Pylons/Units during reward/obs calls
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Robust Error Detection and Bot Safeguards' (Protocol in workflow.md)

## Phase 2: Final State Logic and Reward Attribution
Ensure the winner is correctly identified and rewarded, and the last valid observation is returned.

- [x] Task: Write failing tests for winner identification and final reward values
- [x] Task: Implement `CaptureTheFlagEnv._determine_winner()` and final reward injection
- [x] Task: Implement observation latching (cache last valid frame) in `CaptureTheFlagBot`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Final State and Rewards' (Protocol in workflow.md)

## Phase 3: Integration and Pipeline Stability
Verify the fix in the real training loop.

- [~] Task: Execute `train_pipeline.py` and verify stable transition from episode end to reset
- [ ] Task: Verify final rewards are correctly logged/visible in diagnostic output
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Pipeline Stability' (Protocol in workflow.md)
