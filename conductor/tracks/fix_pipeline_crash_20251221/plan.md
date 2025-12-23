# Track Plan: Fix Training Pipeline Crash (aiohttp & Shape Mismatch)

## Phase 1: Analysis & Reproduction
*   [x] Task: Reproduce the crash. <!-- id: 0 -->
    *   *Goal:* Run `train_pipeline.py` (or a minimal reproduction script) to confirm the exact stack trace and conditions for the crash.
*   [x] Task: Analyze `aiohttp` lifecycle. <!-- id: 1 -->
    *   *Goal:* Trace where `aiohttp.ClientSession` is created and destroyed within the SC2 environment/client code.
*   [x] Task: Analyze Action Mask shape logic. <!-- id: 2 -->
    *   *Goal:* Determine why the mask shape is `32768` (possibly 32 * 1024?) while the expected shape is `[2, 1024]`.
*   [x] Task: Conductor - User Manual Verification 'Phase 1: Analysis & Reproduction' (Protocol in workflow.md)

## Phase 2: Fix Implementation (Test-Driven)
*   [x] Task: Create/Update Tests for Environment Cleanup. <!-- id: 3 -->
    *   *Goal:* Add a test case that initializes and closes the environment repeatedly to simulate the `aiohttp` crash condition.
*   [x] Task: Fix `aiohttp` / Event Loop handling. <!-- id: 4 -->
    *   *Goal:* Implement proper cleanup (e.g., `await session.close()`) in the environment's `close()` method or relevant destructor. Ensure the event loop remains open until cleanup is done.
*   [x] Task: Create/Update Tests for Action Mask Shapes. <!-- id: 5 -->
    *   *Goal:* Add a test ensuring `env.step()` returns action masks of the correct shape `(num_envs, action_dim)` (or individual `(action_dim,)`) under multiprocessing.
*   [x] Task: Fix Action Mask Shape Mismatch. <!-- id: 6 -->
    *   *Goal:* Correct the reshaping or stacking logic for action masks. Ensure the environment returns masks with the correct dimensions.
*   [x] Task: Conductor - User Manual Verification 'Phase 2: Fix Implementation (Test-Driven)' (Protocol in workflow.md)

## Phase 3: Validation
*   [x] Task: Verify fix with short training run. <!-- id: 7 -->
    *   *Goal:* Run the training pipeline for a few iterations to ensure no crashes occur.
*   [x] Task: Verify clean shutdown. <!-- id: 8 -->
    *   *Goal:* Ensure no error messages appear upon script termination.
*   [x] Task: Conductor - User Manual Verification 'Phase 3: Validation' (Protocol in workflow.md)