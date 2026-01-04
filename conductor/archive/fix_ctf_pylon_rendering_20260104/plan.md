# Plan - Fix CTF Pylon Rendering

This plan addresses the visibility and rendering gaps for Pylons in the CTF environment, ensuring they are always visible on the objective layer and correctly represented on unit-based layers.

## Phase 1: Logic Implementation

- [x] Task: Enable `debug_show_map()` for both clients in `train/envs/capture_the_flag_env.py` initialization. (95e1c2c)
- [x] Task: Update `src/transformers/observation_transformer.py` to include `structures` and `enemy_structures` in the unit rendering loop within the `transform` method. (95e1c2c)
- [x] Task: Ensure Pylon radius is correctly handled in `_process_single_unit` and its sub-methods. (95e1c2c)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Logic Implementation' (Protocol in workflow.md) (95e1c2c)

## Phase 2: Verification and Testing

- [x] Task: Create `tests/test_ctf_pylon_rendering.py` to assert Pylon presence on multiple layers (Density, Health, Objective) in a mock environment. (d749a56)
- [x] Task: Verify that Pylons are rendered with the correct intensity and scale. (d749a56)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Verification and Testing' (Protocol in workflow.md) (d749a56)
