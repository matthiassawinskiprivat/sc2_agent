# Plan: Randomized CTF Pylon Spawning

## Phase 1: Test Verification of Spawning Logic [checkpoint: 05ca4e8]
Establish a test case that reproduces the "extra pylon" bug and fails when Pylon locations are static or violate the distance constraint.

- [x] Task: Create `tests/test_ctf_spawning_randomization.py` to check Pylon counts, random positions, and minimum distance. (05ca4e8)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Test Verification of Spawning Logic' (Protocol in workflow.md)

## Phase 2: Implement Randomized Spawning and Cleanup [checkpoint: 05ca4e8]
Refactor `CaptureTheFlagSingleEnv` to implement the new spawning requirements.

- [x] Task: Update `_spawn_initial_units` to implement a "clear map" step (destroy existing units/structures). (05ca4e8)
- [x] Task: Implement random coordinate generation with `flag_distance_min` validation. (05ca4e8)
- [x] Task: Refactor the `debug_create_unit` calls to use the new randomized coordinates for Pylons and units. (05ca4e8)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implement Randomized Spawning and Cleanup' (Protocol in workflow.md)

## Phase 3: Integration and Stability
Verify that the new spawning logic works reliably over multiple episodes without creating invalid states.

- [ ] Task: Run a 100-episode loop (fast reset) to verify 100% compliance with Pylon count and distance constraints.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Stability' (Protocol in workflow.md)
