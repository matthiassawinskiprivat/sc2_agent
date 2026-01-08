# Plan: Fix CTF Multi-Client Hang

This plan outlines the steps to resolve the synchronization hang and connection issues in the CTF environment.

## Phase 1: Implementation & Verification [checkpoint: current]
Fix the stepping logic, bot wiring, and connection stability.

- [x] Task: Implement a synchronized step helper in `CaptureTheFlagEnv`.
- [x] Task: Update `_spawn_initial_units` to use synchronized stepping.
- [x] Task: Switch back to parallel joining with `asyncio.gather` for proper `Portconfig` handshake.
- [x] Task: Properly wire bots using `BotAI._prepare_start` in `_setup_game_session`.
- [x] Task: Fix Windows-specific crash by removing `render=True` from `SC2Process`.
- [x] Task: Fix `AttributeError` in `CaptureTheFlagBot` initialization by populating internal collections.
- [x] Task: Implement aggressive kill-switch in `MultiClientSC2Manager` to prevent zombie processes.
- [x] Task: Fix `ValueError` in `MaskablePPO` by correctly wrapping with `ActionMasker` and using `MaskableSyncVecEnv`.
- [~] Task: Verify fix with a full training run attempt.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Implementation & Verification' (Protocol in workflow.md)
