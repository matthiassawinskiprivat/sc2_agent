# Specification: Fix Action Mask Shape Mismatch in Level 7

## Overview
The training pipeline crashes with a `RuntimeError: shape '[4, 32768]' is invalid for input of size 65536` when running Level 7. This error indicates a discrepancy between the generated action mask and the expanded action space required for the Level 7 environment (Move + Attack). The current mask generation logic produces a mask for a single action type (Move), while the model expects a mask covering both Move and Attack actions.

## Functional Requirements
1.  **Update Mask Generation:**
    -   Modify `Level7Bot` (or the underlying `UnitControlAgent`/`ActionMasker`) to generate an action mask of size `2 * GridSize * GridSize`.
    -   The first half of the mask must correspond to "Move" actions (existing logic).
    -   The second half of the mask must correspond to "Attack" actions.
2.  **Environment Validation:**
    -   Update `Level7Env` to validate that the action mask returned by the bot matches the expected `action_space.n` before passing it to the agent.
    -   Raise a descriptive error if a mismatch is detected.

## Technical Details
-   **Current Behavior:** Mask size is `GridSize^2` (e.g., 16384 for 128x128).
-   **Target Behavior:** Mask size must be `2 * GridSize^2` (e.g., 32768).
-   **Error Analysis:** `65536` (Input Size) / `4` (Batch) = `16384` (Single Mask). The model expects `32768` per sample.
-   **Attack Mask Logic:** For this fix, the attack mask can default to being fully open (valid) or identical to the move mask, unless specific attack-range logic is already available to be integrated.

## Acceptance Criteria
-   [ ] `train_pipeline.py` runs for Level 7 without crashing due to shape mismatch.
-   [ ] The action mask passed to the model has the correct shape `(Batch, 2 * GridSize^2)`.
-   [ ] `test_action_mask_shape` (new or existing test) passes.
