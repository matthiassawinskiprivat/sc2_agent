# Plan - Fix premature train_pipeline termination/hang

## Phase 1: Diagnostic Instrumentation & Observability [checkpoint: 5ff509f]
- [x] Task: Add high-resolution logging to `train/game_process.py` (5ff509f)
    - [x] Write failing test to verify logging output during process lifecycle
    - [x] Implement detailed logging for SC2 process creation, monitoring, and termination
    - [x] Verify logging in test environment
- [x] Task: Add detailed instrumentation to `train/sync_vec_env.py` (5ff509f)
    - [x] Write failing test to verify logging during `step` and `reset` synchronization
    - [x] Implement logging around inter-process communication (queues/pipes)
    - [x] Verify logging captures state transitions
- [x] Task: Enhance `train/train_pipeline.py` with loop diagnostics (5ff509f)
    - [x] Write failing test to verify logging of iteration counts and exit conditions
    - [x] Implement logging for session start, soft/hard resets, and termination checks
    - [x] Verify logging output
- [x] Task: Conductor - User Manual Verification 'Diagnostic Instrumentation & Observability' (Protocol in workflow.md) (5ff509f)

## Phase 2: Watchdog Implementation
- [ ] Task: Develop a generic Watchdog utility
    - [ ] Write failing tests for a watchdog that triggers a callback on timeout
    - [ ] Implement `Watchdog` class with start/stop/reset and stack trace dumping
    - [ ] Verify watchdog triggers correctly in isolation
- [ ] Task: Integrate Watchdog into the training loop
    - [ ] Write failing test where a simulated hang triggers the watchdog
    - [ ] Integrate watchdog into `train_pipeline.py` to monitor the main training loop
    - [ ] Verify watchdog detects simulated hangs in the pipeline
- [ ] Task: Conductor - User Manual Verification 'Watchdog Implementation' (Protocol in workflow.md)

## Phase 3: Root Cause Analysis & Resolution
- [ ] Task: Reproduce hang and analyze diagnostic data
    - [ ] Run the training pipeline until a hang occurs
    - [ ] Capture and analyze logs and watchdog stack traces to identify the hang point
- [ ] Task: Implement fix for identified hang/deadlock
    - [ ] Write a regression test that reproduces the hang condition (if possible) or validates the fix logic
    - [ ] Apply the fix to the identified component (e.g., communication timeout, process join lock)
    - [ ] Verify the fix resolves the hang in the test environment
- [ ] Task: Strengthen StarCraft 2 process cleanup
    - [ ] Write failing test to verify orphaned SC2 processes are detected and killed
    - [ ] Implement robust cleanup logic in `game_process.py` (e.g., using `psutil` to ensure all children are gone)
    - [ ] Verify cleanup works even after a pipeline hang/crash
- [ ] Task: Conductor - User Manual Verification 'Root Cause Analysis & Resolution' (Protocol in workflow.md)
