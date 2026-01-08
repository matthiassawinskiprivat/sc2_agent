# Plan - Fix CTF Pylon Coordinate Offsets

This plan addresses the coordinate system mismatch where relative Pylon positions were used as absolute coordinates, preventing spawning in the top-right areas of the map.

## Phase 1: Environment Logic Update

- [x] Task: Update `train/envs/capture_the_flag_single_env.py` to add `playable_area.x` and `playable_area.y` offsets to the randomized Pylon positions. (2794eac)
- [x] Task: Update `train/envs/capture_the_flag_env.py` to add `playable_area.x` and `playable_area.y` offsets to the randomized Pylon positions. (2794eac)
- [x] Task: Reduce the hardcoded margin in both environments to 1.0 or 2.0 to maximize usable space. (2794eac)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Environment Logic Update' (Protocol in workflow.md) (2794eac)

## Phase 2: Verification

- [x] Task: Update `tests/verify_ctf_randomization.py` to assert that Pylon positions cover the full range of the `playable_area` (including values near the top-right limits). (e112b46)
- [x] Task: Run the updated verification script and confirm that spawning is now distributed across the entire map. (e112b46)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Verification' (Protocol in workflow.md) (e112b46)
