# Implementation Plan - Fix Training Freeze in `is_image_space`

## Phase 1: Preparation & Testing Strategy [checkpoint: 766f5e1]
- [x] Task: Create a reproduction test `tests/test_obs_space_check_performance.py`.
    - [x] Instantiate the environment (e.g., `Level0Env`) and retrieve its `observation_space`.
    - [x] Simulate the Stable Baselines 3 check: `from stable_baselines3.common.preprocessing import is_image_space; is_image_space(env.observation_space)`.
    - [x] Measure execution time and assert it finishes in under 50ms (Expect it to hang or be extremely slow currently).
- [x] Task: Run the test to confirm the performance bottleneck (Red Phase).
- [x] Task: Conductor - User Manual Verification 'Preparation & Testing Strategy' (Protocol in workflow.md)

## Phase 2: Implementation
- [ ] Task: Refactor the `observation_space` definition.
    - [ ] Locate the space definition in `src/transformers/observation_transformer.py` or the environment wrappers.
    - [ ] Ensure `Box` spaces for image channels use scalar bounds (e.g., `low=0, high=255`) rather than `np.full(...)` arrays.
    - [ ] Ensure `dtype=np.uint8` is strictly used for all spatial/image layers.
- [ ] Task: Run the performance test `tests/test_obs_space_check_performance.py` to verify the fix (Green Phase).
- [ ] Task: Conductor - User Manual Verification 'Implementation' (Protocol in workflow.md)

## Phase 3: Validation & Cleanup
- [ ] Task: Run a sanity training pass (minimal steps) using `train_pipeline.py` to ensure the watchdog no longer triggers.
- [ ] Task: Conductor - User Manual Verification 'Validation & Cleanup' (Protocol in workflow.md)
