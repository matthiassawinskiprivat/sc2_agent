# Plan: Dynamic Input Dimension Transfer (Weight Surgery)

## Phase 1: Model Surgery Utility [checkpoint: 468a1f9]
*Goal: Implement a robust utility to handle weight grafting for PyTorch CNNs.*

- [x] Task: Create `src/utils/model_utils.py` and implement weight grafting logic for input layer expansion.
- [x] Task: Write Failing Tests (Red Phase) for `graft_model_weights` verifying channel preservation.
- [x] Task: Implement `graft_model_weights` to pass tests (Green Phase).
- [x] Task: Refactor and ensure >80% coverage for the utility.
- [x] Task: Conductor - User Manual Verification 'Model Surgery Utility' (Protocol in workflow.md)

## Phase 2: Pipeline Integration & Verification [checkpoint: efe088a]
*Goal: Integrate the utility into the training pipeline and verify end-to-step curriculum loading.*

- [x] Task: Update `PipelineManager._train_level` in `train_pipeline.py` to detect and handle `observation_space` mismatches.
- [x] Task: Update `PipelineManager` to handle simultaneous observation and action space transitions.
- [x] Task: Write Integration Test simulating a full level transition (L0 model -> L1 environment).
- [x] Task: Verify the entire loading flow with a dummy model surgery.
- [x] Task: Conductor - User Manual Verification 'Pipeline Integration & Verification' (Protocol in workflow.md)
