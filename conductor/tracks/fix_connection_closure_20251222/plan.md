# Track Plan: Fix Connection Closure in Training Pipeline

## Phase 1: Diagnostics & Targeted Testing
*   [x] Task: Create a high-frequency reset stress test. <!-- id: 0 -->
    *   *Goal:* Isolate the `WSMessageTypeError` by performing 50+ rapid environment resets with map rotations.
    *   *Context:* Create `tests/test_reset_stress.py`.
*   [x] Task: Trace `burnysc2` protocol reception logic. <!-- id: 1 -->
    *   *Goal:* Add temporary logging to `venv/Lib/site-packages/sc2/protocol.py` to capture the raw WebSocket messages before the `WSMessageTypeError` occurs.
    *   *Context:* Identify if the client is sending a `CLOSE` frame or something else.
*   [x] Task: Conductor - User Manual Verification 'Phase 1: Diagnostics & Targeted Testing' (Protocol in workflow.md)

## Phase 2: Stabilization & Robustness (Test-Driven)
*   [x] Task: Implement `WSMessageTypeError` handling in `Level0Env`. <!-- id: 2 -->
    *   *Goal:* Catch the error during `reset()` and trigger a full `close()` and re-initialization of the `SC2Process`.
    *   *Context:* Modify `train/envs/level_0_env.py`. Write a test that mocks `aiohttp` to throw this error and verifies the environment recovers.
*   [x] Task: Refactor "Hard Reset" Atomicity. <!-- id: 3 -->
    *   *Goal:* Ensure `self.close()` is fully awaited and all resources are nullified before starting a new `SC2Process`.
    *   *Context:* Tighten the logic in `Level0Env.reset()` and `Level0Env.close()`.
*   [~] Task: Fix Race Condition in `join_game`. <!-- id: 4 -->
    *   *Goal:* Add a small, configurable delay or a "ready check" before attempting `join_game` after a process launch.
*   [ ] Task: Conductor - User Manual Verification 'Phase 2: Stabilization & Robustness (Test-Driven)' (Protocol in workflow.md)

## Phase 3: Final Validation
*   [ ] Task: Verify long-term stability. <!-- id: 5 -->
    *   *Goal:* Run `train_pipeline.py` for 50,000 timesteps across all levels to ensure no connection drops.
*   [ ] Task: Verify resource cleanup. <!-- id: 6 -->
    *   *Goal:* Confirm no `SC2.exe` processes remain after a pipeline crash or completion.
*   [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Validation' (Protocol in workflow.md)
