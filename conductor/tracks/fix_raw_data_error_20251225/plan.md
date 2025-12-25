# Plan: Fix AttributeError: raw_data in Training Pipeline

## Phase 1: Diagnosis and Reproduction
- [x] Task: Create a minimal reproduction script for the `AttributeError: raw_data`
    - [x] Create `reproduce_raw_data_error.py` that mimics the `level_0_env.py` reset logic.
    - [x] Run the script and confirm it triggers the exact same `AttributeError`.
- [x] Task: Inspect `sc2.game_state.GameState` and `ResponseObservation` structure
    - [x] Use `inspect_observation.py` (or similar) to print the attributes of the object returned by `client.observation()`.
    - [x] Compare findings with `sc2` library source code to identify where the mismatch occurs.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Diagnosis and Reproduction' (Protocol in workflow.md)

## Phase 2: Fix Implementation
- [x] Task: Implement fix in `train/envs/level_0_env.py`
    - [x] Update `_setup_game_session` and `_reset_bot_state` to handle the observation object correctly.
    - [x] Ensure `ProtoWrapper` or similar mechanisms are used if required by the library version.
- [x] Task: Verify fix with reproduction script
    - [x] Run `reproduce_raw_data_error.py` and confirm the `AttributeError` is resolved.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Fix Implementation' (Protocol in workflow.md)

## Phase 3: Integration and Regression Testing
- [x] Task: Run existing environment tests
    - [x] Run `pytest tests/test_action_mask_validity.py`.
    - [x] Run `pytest tests/test_robustness_and_init.py`.
- [~] Task: Verify full `train_pipeline`
    - [ ] Run `python -m train.train_pipeline` for a short duration (e.g., 1000 steps).
    - [ ] Confirm no crashes occur during multiple resets.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Regression Testing' (Protocol in workflow.md)
