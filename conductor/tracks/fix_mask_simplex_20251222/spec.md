# Track Specification: Fix Action Masking Simplex Error

## 1. Overview
The training pipeline crashes shortly after initialization with a `ValueError: Expected parameter probs ... to satisfy the constraint Simplex()`. This indicates that the probabilities produced by `MaskablePPO` do not sum to 1.0, typically because all actions are being masked out or the masks are being passed in an incorrect format by the recently implemented `MaskableSubprocVecEnv`.

## 2. Functional Requirements
*   **Audit `MaskableSubprocVecEnv`:** Verify that `get_action_masks()` returns a boolean NumPy array of shape `(num_envs, action_dim)` where each row contains at least one `True` value.
*   **Validate Mask Content:** Implement a safety check in `Level0Env.action_masks()` to ensure that the environment never returns an empty mask (all `False`), which would break the distribution.
*   **Verify Bot Initialization:** Ensure that the `HierarchicalAgentSystem` and its unit agents are fully initialized and have processed at least one valid observation before the first mask is requested.
*   **Regression Check:** Ensure the fix for the `aiohttp` connection closure remains stable during these changes.

## 3. Non-Functional Requirements
*   **Logging:** Add debug logging to capture the sum of the mask and the number of valid actions during the first 100 steps of training.
*   **Type Safety:** Ensure the masks are strictly boolean tensors before they reach the PyTorch distribution.

## 4. Acceptance Criteria
*   [ ] The `train_pipeline.py` script can complete a short training run (e.g., 2048 steps) without a `ValueError`.
*   [ ] `tests/test_action_mask_validity.py` verifies that masks always contain valid actions across multiple resets and maps.
*   [ ] Automated tests confirm that `MaskableSubprocVecEnv` produces the exact format required by `sb3-contrib`.

## 5. Out of Scope
*   Optimizing the CNN architecture or feature extraction logic.
*   Fixing unrelated UI or visualization bugs.
