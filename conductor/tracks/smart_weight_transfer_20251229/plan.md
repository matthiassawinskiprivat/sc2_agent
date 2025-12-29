# Plan: Smart Weight Transfer

## Phase 1: Core Interpolation Logic
- [x] Task: Create `tests/test_weight_interpolation.py` with failing tests for upscaling a mock weight tensor (32x32 -> 64x64) using bilinear interpolation. (32ee77b)
- [x] Task: Implement `interpolate_spatial_weights` in `src/utils/model_utils.py` that uses `torch.nn.functional.interpolate` to resize weights while preserving mean/variance. (32ee77b)
- [~] Task: Verify tests pass and document coverage.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Interpolation Logic' (Protocol in workflow.md)

## Phase 2: Action Head Adaptation
- [ ] Task: Create `tests/test_action_head_grafting.py` to verify that a full `state_dict` can have its action heads (e.g., `action_net.weight`) upscaled.
- [ ] Task: Implement `upscale_action_heads` in `src/utils/model_utils.py` to identify and resize all grid-dependent layers in the MaskablePPO policy.
- [ ] Task: Verify tests pass and document coverage.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Action Head Adaptation' (Protocol in workflow.md)

## Phase 3: Pipeline Integration
- [ ] Task: Create integration test `tests/test_pipeline_weight_transfer.py` that simulates loading a 32x32 model into a 64x64 environment.
- [ ] Task: Update `PipelineManager._train_level` in `train/train_pipeline.py` to detect grid size mismatches and invoke the weight transfer logic.
- [ ] Task: Verify tests pass and document coverage.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Pipeline Integration' (Protocol in workflow.md)

## Phase 4: Final Verification
- [ ] Task: Run a manual test training for 1000 steps using an upscaled model to ensure `ep_len_mean` starts at a reasonable value (indicating preserved knowledge).
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Verification' (Protocol in workflow.md)
