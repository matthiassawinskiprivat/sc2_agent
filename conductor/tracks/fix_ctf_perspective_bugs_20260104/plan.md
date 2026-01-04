# Plan - Fix CTF Perspective Reward and Target Bugs

## Phase 1: Logic Implementation

- [x] Task: Refactor `CaptureTheFlagBot` to use per-player state for `command_targets` and `prev_dist_to_pylon`. (ab4dc1d)
- [x] Task: Update `calculate_reward` to be fully perspective-aware by implementing structure flipping. (ab4dc1d)
- [x] Task: Update `get_observation` and `apply_action` to use per-player state. (ab4dc1d)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Logic Implementation' (Protocol in workflow.md) (ab4dc1d)

## Phase 2: Verification

- [x] Task: Run `debug_ctf_visuals.py` and verify symmetric objective placement. (ab4dc1d)
- [x] Task: Verify rewards are correctly calculated for both players (positive for moving towards enemy pylon). (ab4dc1d)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Verification' (Protocol in workflow.md) (ab4dc1d)
