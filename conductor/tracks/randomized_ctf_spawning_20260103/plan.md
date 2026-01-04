# Plan - Randomized CTF Pylon Spawning

This plan focuses on implementing robust, consistent randomization for Pylon spawning across both the single-client and multi-client CTF environments.

## Phase 1: Shared Randomization Logic

- [x] Task: Extract or synchronize the `get_rand_pos` and distance check logic to ensure parity between `CaptureTheFlagEnv` and `CaptureTheFlagSingleEnv`. (82dee25)
- [x] Task: Implement a scaling distance check that defaults to a percentage of the map diagonal if `flag_distance_min` is not provided. (82dee25)
- [x] Task: Write unit tests to verify the coordinate generation logic stays within map bounds and respects margins. (82dee25)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Shared Randomization Logic' (Protocol in workflow.md) (82dee25)

## Phase 2: Environment Implementation

- [ ] Task: Update `train/envs/capture_the_flag_env.py` to replace hardcoded Pylon coordinates with the new randomization logic.
- [ ] Task: Refine `train/envs/capture_the_flag_single_env.py` logic to use the same robust method and ensure its coordinate rounding doesn't cause grid snapping issues.
- [ ] Task: Add detailed `loguru` logging to both environments to report Pylon positions and distances during every `_spawn_initial_units` call.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Environment Implementation' (Protocol in workflow.md)

## Phase 3: Verification

- [ ] Task: Create a reproduction/verification script that resets the environments multiple times and asserts that Pylon positions vary significantly.
- [ ] Task: Verify that the `flag_distance_min` constraint is strictly honored in both environments.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Verification' (Protocol in workflow.md)