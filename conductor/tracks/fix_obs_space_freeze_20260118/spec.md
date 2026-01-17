# Specification: Fix Training Freeze in `is_image_space`

## Overview
The training pipeline freezes or hangs in the `is_image_space` function within Stable Baselines 3's preprocessing logic. This function checks if the observation space bounds are exactly 0 and 255. If the `observation_space` is defined with large array bounds (e.g., per-pixel bounds) instead of scalars, or if the `dtype` is slightly off, this check can become computationally expensive or hang, especially with large observation grids.

This track will optimize the `observation_space` definition in `src/transformers/observation_transformer.py` (or equivalent) to ensure it is defined efficiently, preventing this freeze.

## Functional Requirements
1.  **Efficient Observation Space:** The `observation_space` for image-like inputs (CNN grids) must be defined using scalar bounds (`low=0`, `high=255`) where possible, or `np.uint8` types that SB3 can handle efficiently.
2.  **SB3 Compatibility:** The defined space must pass `is_image_space` checks without performance degradation.

## Non-Functional Requirements
1.  **No Logic Change:** The actual content of the observations must not change; only the metadata (Space definition) describing them should be optimized.
2.  **Training Stability:** The fix should prevent the watchdog timeout and stack trace dumping currently observed.

## Acceptance Criteria
1.  The `is_image_space` check completes instantly.
2.  Training starts and proceeds without the reported watchdog timeout in `train_pipeline.py`.
3.  The observation space properties (`shape`, `dtype`, `low`, `high`) are verified to be correct.

## Out of Scope
*   Disabling `normalize_images` globally (unless the space optimization proves impossible, but the goal here is to fix the space definition).
*   Changing the neural network architecture.
