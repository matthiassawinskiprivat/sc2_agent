# Plan: Speed Up CTF Environment Start-Up

This plan outlines the steps to optimize the start-up sequence of the Capture the Flag environment by auditing delays and streamlining the handshake process.

## Phase 1: Performance Profiling and Baseline
Establish a precise measurement of where the time is spent during start-up.

- [x] Task: Create a profiling script to measure time for each sub-step of `reset()`
- [x] Task: Establish a baseline start-up time and identify the top 3 delay-heavy operations
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Performance Profiling and Baseline' (Protocol in workflow.md)

## Phase 2: Delay Optimization and Manager Refinement
Audit and reduce hardcoded delays in the Multi-Client Manager and Environment.

- [x] Task: Write regression tests to verify stability during rapid reset cycles
- [x] Task: Audit and minimize `asyncio.sleep` calls in `MultiClientSC2Manager.launch_processes`
- [x] Task: Audit and minimize delays in `CaptureTheFlagEnv._setup_game_session` and spawning logic
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Delay Optimization and Manager Refinement' (Protocol in workflow.md)

## Phase 3: Final Verification and Stress Testing
Ensure the optimizations don't break stability over many cycles.

- [~] Task: Run stress test (50 consecutive resets) to confirm zero regression in stability
- [ ] Task: Compare final start-up time against baseline and document gains
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Verification and Stress Testing' (Protocol in workflow.md)
