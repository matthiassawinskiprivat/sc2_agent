# Track Specification: Fix Training Pipeline Crash (aiohttp & Shape Mismatch)

## 1. Overview
The training pipeline (`train_pipeline.py`) is failing with two distinct errors:
1.  **`aiohttp` / Event Loop Error:** An `Exception ignored in: <function ClientResponse.__del__>` followed by `RuntimeError: Event loop is closed`. This indicates improper cleanup of async resources, likely within the SC2 client or environment during process teardown.
2.  **Shape Mismatch in PPO Mask:** A `RuntimeError: shape '[2, 1024]' is invalid for input of size 32768` occurring during `model.learn`. This suggests that the action masks provided by the environment do not match the expected dimensions of the policy's action distribution, possibly exacerbated by the recent multiprocessing changes.

The goal is to perform a robust stabilization of the pipeline to resolve these crashes and ensure reliable training.

## 2. Functional Requirements
*   **Fix `aiohttp` Cleanup:** Identify and properly close `aiohttp` sessions/connectors when environments or the SC2 client shut down, preventing "Event loop is closed" errors.
*   **Resolve Shape Mismatch:** Investigate the action mask generation and passing mechanism. Ensure that the shape of `action_masks` returned by the environment matches `(n_envs, action_space_size)` (or the specific flattened dimension expected by the policy) at all times.
*   **Validate Multiprocessing:** Ensure that the `SubprocVecEnv` (or equivalent) correctly handles environment resets and step returns without corrupting data structures like action masks.

## 3. Non-Functional Requirements
*   **Stability:** The training pipeline must run without crashing for extended periods.
*   **Clean Shutdown:** The system should exit gracefully without leaving zombie processes or unclosed resources.

## 4. Acceptance Criteria
*   [ ] The `train_pipeline.py` script runs to completion (or at least through multiple updates) without raising `RuntimeError: Event loop is closed`.
*   [ ] The `RuntimeError: shape '...' is invalid` error is resolved, and the policy can successfully sample actions using the provided masks.
*   [ ] A test run of the training pipeline (e.g., using `debug_train.py` or a short run of the main pipeline) passes successfully.

## 5. Out of Scope
*   Refactoring the entire reinforcement learning algorithm (PPO) logic, unless directly related to the masking interface.
*   Major feature additions to the agent's gameplay logic.
