# Specification: Smart Weight Transfer

## Overview
This feature implements a mechanism to adapt a trained agent's policy when the spatial action resolution (grid size) changes (e.g., from 32x32 to 64x64). This preserves learned behaviors (spatial knowledge) instead of resetting the action head to random values, significantly reducing relearning time between curriculum levels.

## Functional Requirements
- **Generic Scaling Function:** Implement a transfer function that supports arbitrary grid size transitions (e.g., 32 -> 64, 64 -> 128).
- **Spatial Interpolation:** Use bilinear interpolation to resize action head weights (and value heads if spatial) to match the new output dimensions.
- **Integration:** The `PipelineManager` in `train/train_pipeline.py` should automatically detect grid size mismatches and apply this transfer during model loading.
- **State Dictionary Management:** Ensure the `state_dict` is correctly manipulated so that Stable Baselines 3 can load the "grafted" model without errors.

## Non-Functional Requirements
- **Numerical Stability:** Preserve the mean and variance of weights during interpolation to avoid destabilizing the initial steps of training after transfer.
- **Maintainability:** Modularize the logic within `src/utils/model_utils.py`.

## Acceptance Criteria
- [ ] A unit test proves that an agent trained on 32x32 to "move right" still predominantly "moves right" after being upscaled to 64x64.
- [ ] The `train_pipeline.py` can load a Level 0 (32x32) model into a Level 1 (64x64) environment without crashing or discarding all learned knowledge.
- [ ] Verification script/logs show that weight statistics remain stable after transfer.

## Out of Scope
- Modifying the CNN feature extractor (handled by existing `graft_model_weights`).
- Changing the underlying RL algorithm parameters (PPO).
