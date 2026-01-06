# Plan: Fix Parallel Process Termination

This plan resolves the parallel training crash by ensuring per-episode environment resets only terminate their own StarCraft II instances.

## Phase 1: Localize Cleanup Logic
Refactor the manager to be process-aware during cleanup.

- [x] Task: TDD - Create a test that verifies `SingleClientSC2Manager.close()` terminates its own process without killing a "sibling" SC2 process. d5a20e0
- [x] Task: Remove `kill_switch_sc2()` from `SingleClientSC2Manager.close()`. d5a20e0
- [x] Task: Implement targeted process killing in `SingleClientSC2Manager.close()` using the stored process PID. d5a20e0
- [x] Task: Conductor - User Manual Verification 'Phase 1: Localize Cleanup Logic' (Protocol in workflow.md) d5a20e0

## Phase 2: Parallel Stress Test
Verify stability across multiple episodes in parallel.

- [x] Task: Integration Test - Run `tests/test_ctf_num_envs_scaling.py` (or similar) and verify it can complete at least 2 full resets across multiple workers. d5a20e0
- [x] Task: Conductor - User Manual Verification 'Phase 2: Parallel Stress Test' (Protocol in workflow.md) d5a20e0