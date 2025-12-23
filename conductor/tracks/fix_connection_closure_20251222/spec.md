# Track Specification: Fix Connection Closure in Training Pipeline

## 1. Overview
The training pipeline crashes during the `reset()` sequence of `Level0Env`, specifically at the `join_game` step. The root cause is a `WSMessageTypeError` (Type 257, which corresponds to `WSMsgType.CLOSED`) raised by `aiohttp` within the `burnysc2` protocol handler. This indicates the WebSocket connection to the StarCraft II client is being closed unexpectedly, possibly due to a race condition or improper cleanup during the recently implemented "hard reset" and map rotation logic.

## 2. Functional Requirements
*   **Investigate `burnysc2` Protocol Interaction:** Analyze how the `Protocol` class in `burnysc2` handles WebSocket messages and why it's receiving a `CLOSED` type during an active request (`join_game`).
*   **Stabilize Hard Reset Logic:** Ensure that the "hard reset" mechanism (closing and re-opening the SC2 process) in `Level0Env` is atomic and that all resources (loops, clients, processes) are fully cleared before re-initialization.
*   **Robust Connection Recovery:** Implement error handling to catch `ConnectionAlreadyClosed` or `WSMessageTypeError` during the setup phase, allowing the environment to retry the launch/join sequence instead of crashing the entire pipeline.
*   **Action Mask Consistency:** Verify that the action mask shape remains consistent (as fixed in the previous track) even after multiple hard resets and map changes.

## 3. Non-Functional Requirements
*   **Detailed Logging:** Add tracing to the `Level0Env` reset and setup phases to capture the exact state of the process and WebSocket before a failure.
*   **Graceful Teardown:** Ensure the `close()` method is invoked correctly during failures to prevent zombie `SC2.exe` processes.

## 4. Acceptance Criteria
*   [ ] The training pipeline can successfully cycle through multiple "hard resets" and map rotations without a `ConnectionAlreadyClosed` error.
*   [ ] The specific error `aiohttp.client_exceptions.WSMessageTypeError: Received message 257` is handled or prevented.
*   [ ] A test script can perform 20+ consecutive resets with map changes without failure.

## 5. Out of Scope
*   Modifying the core StarCraft II binary or protocol buffers.
*   Changing the underlying RL algorithm (PPO) hyper-parameters.
