# Specification: Dynamic Input Dimension Transfer (Weight Surgery)

## Overview
Currently, adding new observation layers in later curriculum levels causes a shape mismatch error in the first layer of the CNN. This track implements a "Weight Surgery" utility to allow pre-trained models to be loaded into environments with expanded observation spaces by grafting old weights into a larger input layer.

## Functional Requirements
1.  **Observation Space Mismatch Detection**:
    *   Update the `train_pipeline.py` loading logic to detect differences in the `observation_space` channel count between the saved model and the current environment.
2.  **Model Grafting Utility (`src/utils/model_utils.py`)**:
    *   Implement a function to handle weight transfer for PyTorch models (specifically SB3's `CnnPolicy`).
    *   Identify the first convolutional layer in the feature extractor.
    *   Create a new weight tensor with the expanded shape.
    *   **Weight Copying**: Preserve learned weights for existing channels at their original indices.
    *   **New Channel Initialization**: Initialize weights for new channels with small random noise to allow learning without breaking pre-trained features.
3.  **Comprehensive Transfer**:
    *   Ensure the surgery handles cases where *both* observation and action spaces have changed.
    *   Verify that policy and value network weights remain intact.

## Non-Functional Requirements
*   **Safety**: Ensure the original model file is never overwritten by the modified version unless explicitly saved as a new checkpoint.
*   **Modularity**: Keep the surgery logic decoupled from the pipeline in a utility module.

## Acceptance Criteria
*   [ ] `src/utils/model_utils.py` is created with a robust grafting function.
*   [ ] `train_pipeline.py` successfully loads a "Level 0" (4-layer) model into a "Level 1" (e.g., 5-layer) environment without errors.
*   [ ] Unit tests verify that weights for index 0-3 are identical after transfer.
*   [ ] Integration tests verify the model can still perform a forward pass (inference) post-surgery.

## Out of Scope
*   Handling architecture changes other than input/output widths.
*   Optimizing the surgery for non-CNN policies.
