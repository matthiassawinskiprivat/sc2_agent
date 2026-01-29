# Specification - Fix LevelDrop Model Mismatch

## Overview
The training pipeline failed when transitioning from `CaptureTheFlag` to `LevelDrop` due to architectural inconsistencies:
1. `net_arch` mismatch ([64, 64] vs []).
2. `grid_size` mismatch (128 vs 10).
3. `action_space` mismatch (32768 vs 400).

The agent should support adding new action heads (`load`, `unload`) to the existing ones (`move`, `attack`) while maintaining the 128x128 spatial resolution.

## Functional Requirements
- `LevelDropEnv` must initialize with 4 action heads: `move`, `attack`, `load`, and `unload`.
- `LevelDropEnv` must use a `grid_size` of 128 to match the resolution of pre-trained models.
- The training configuration must explicitly set `net_arch: []` to avoid SB3 defaults.

## Technical Details
- Update `config/level_drop_0.yaml` with the correct `grid_size` and `net_arch`.
- Refactor `LevelDropEnv` to use cumulative action head resolution.
- Verify that `graft_model_weights` performs 2x expansion for action heads.

## Acceptance Criteria
- `train_pipeline.py` successfully loads the CTF model for LevelDrop.
- `action_space.n` is confirmed to be 65536.
- Training starts and progresses for at least one iteration.
