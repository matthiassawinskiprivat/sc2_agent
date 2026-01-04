# Plan - Randomized CTF Pylon Spawning

This plan focuses on implementing robust, consistent randomization for Pylon spawning across both the single-client and multi-client CTF environments.

## Phase 1: Shared Randomization Logic

- [x] Task: Extract or synchronize the `get_rand_pos` and distance check logic to ensure parity between `CaptureTheFlagEnv` and `CaptureTheFlagSingleEnv`. (82dee25)
- [x] Task: Implement a scaling distance check that defaults to a percentage of the map diagonal if `flag_distance_min` is not provided. (82dee25)
- [x] Task: Write unit tests to verify the coordinate generation logic stays within map bounds and respects margins. (82dee25)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Shared Randomization Logic' (Protocol in workflow.md) (82dee25)

## Phase 2: Environment Implementation

- [x] Task: Update `train/envs/capture_the_flag_env.py` to replace hardcoded Pylon coordinates with the new randomization logic. (df0af72)
- [x] Task: Refine `train/envs/capture_the_flag_single_env.py` logic to use the same robust method and ensure its coordinate rounding doesn't cause grid snapping issues. (df0af72)
- [x] Task: Add detailed `loguru` logging to both environments to report Pylon positions and distances during every `_spawn_initial_units` call. (df0af72)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Environment Implementation' (Protocol in workflow.md) (df0af72)

## Phase 3: Verification

- [x] Task: Create a reproduction/verification script that resets the environments multiple times and asserts that Pylon positions vary significantly. (2f1536e)
- [x] Task: Verify that the `flag_distance_min` constraint is strictly honored in both environments. (2f1536e)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Verification' (Protocol in workflow.md) (2f1536e)