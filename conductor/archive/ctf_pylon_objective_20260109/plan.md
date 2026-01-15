# Plan: CTF Pylon-Only Objective Rendering

This plan implements exclusive Pylon tracking for the command objective layer in CTF.

## Phase 1: Implementation
- [x] Task: Initialize `enemy_pylon_pos` in `CaptureTheFlagBot.__init__`.
- [x] Task: Update `get_observation` to track and use Pylon position for `command_target`.
- [x] Task: Remove legacy `command_targets` and cleanup `apply_action`.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Implementation' (Protocol in workflow.md)

## Phase 2: Verification
- [x] Task: Run existing `tests/test_ctf_pylon_rendering.py`.
- [x] Task: Create and run `tests/test_ctf_objective_logic.py` to verify unit isolation and persistent tracking.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Verification' (Protocol in workflow.md)
