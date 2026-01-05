# Implementation Plan - Perspective Flipping

## Phase 1: Investigation and Design
- [x] Task: Analyze `ObservationTransformer` and `SharedObservationTransformer` to map all coordinate systems (Pixel vs. Grid vs. World).
- [x] Task: Analyze `UnitControlAgent` to understand how spatial actions are currently mapped to world coordinates.
- [x] Task: Analyze `CaptureTheFlagBot.calculate_reward` to determine if reward logic needs explicit coordinate flipping or if it relies on relative entity distances (which might be safe).
- [x] Task: Design the `PerspectiveTransformer` utility class interface (methods for `flip_observation`, `flip_action`, `flip_mask`).

## Phase 2: Core Implementation (TDD) [checkpoint: 21b19f5]
- [x] Task: Create `tests/test_perspective_transformer.py` with failure cases for flipping observations (numpy arrays) and actions (integer indices).
- [x] Task: Implement `src/utils/perspective_transformer.py`. [8155bb2]
    -   Implement `flip_observation(obs, spatial_dim)`: Rotates/Flips the feature maps.
    -   Implement `flip_action(action, grid_size)`: Converts a spatial action index to its symmetric counterpart.
    -   Implement `flip_mask(mask, grid_size)`: Flips the action mask to match the flipped view.
- [x] Task: Verify `PerspectiveTransformer` with unit tests.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Core Implementation (TDD)' (Protocol in workflow.md)

## Phase 3: Integration [checkpoint: 5106830]
- [x] Task: Create `tests/test_ctf_perspective_integration.py` to verify the full pipeline (Env -> Bot -> Transformer -> Env).
- [x] Task: Integrate `PerspectiveTransformer` into `CaptureTheFlagEnv` (or `CaptureTheFlagBot` depending on analysis).
    -   Apply observation flipping for Player 2 before returning to agent.
    -   Apply action flipping for Player 2 before executing in SC2.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Integration' (Protocol in workflow.md)

## Phase 4: Verification & Cleanup
- [ ] Task: Run `train_pipeline.py` briefly (dry run) to ensure no crashes with the new logic.
- [ ] Task: Manual verification: Watch a replay or debug run to confirm Player 2 moves "forward" (relative to itself) when the agent commands "forward".
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Verification & Cleanup' (Protocol in workflow.md)
