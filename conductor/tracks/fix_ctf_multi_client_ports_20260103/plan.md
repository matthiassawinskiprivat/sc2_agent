# Plan: Fix CTF Multi-Client Ports

This plan outlines the steps to resolve the communication issues between multiple SC2 clients.

## Phase 1: Implementation & Verification
Implement port allocation and logging.

- [x] Task: Add logging to `MultiClientEnv` to capture subprocess crashes.
- [x] Task: Implement `Portconfig` in `MultiClientSC2Manager.create_join_game`.
- [x] Task: Verify fix by running `train_pipeline` for `capture_the_flag`. 51f6d26
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Implementation & Verification' (Protocol in workflow.md)
