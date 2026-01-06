# Plan: Fix Persistent Termination Failure in CTF

This plan targets the root cause of the termination failure by fixing unit property shadowing in the bot and refining the asymmetric termination logic.

## Phase 1: Bot Integrity & Diagnostics
Address the likely root cause: the bot shadowing StarCraft II's dynamic unit properties with static lists.

- [x] Task: TDD - Create a test verifying that `CaptureTheFlagBot.units` reflects changes in the game state rather than staying static. df8d017
- [x] Task: Remove shadowing of `units`, `structures`, etc., in `CaptureTheFlagBot.__init__`. df8d017
- [x] Task: Add diagnostic logging to `CaptureTheFlagEnv._check_termination` to print unit counts and types when a player is close to 0. df8d017
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Bot Integrity & Diagnostics' (Protocol in workflow.md)

## Phase 2: Enhanced Asymmetric Termination
Refine the detection and branching logic to ensure the game ends/continues correctly.

- [x] Task: TDD - Update `_has_combat_units` tests to include "Ghost" units and edge-case entity types. df8d017
- [x] Task: Implement enhanced filtering in `_has_combat_units` (ensuring it uses the live `bot.units` property). df8d017
- [x] Task: Implement corrected termination branching in `_check_termination` (P1 Loss = Done, P2 Loss = Continue). df8d017
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Enhanced Asymmetric Termination' (Protocol in workflow.md)

## Phase 3: Regression & Final Cleanup
Ensure stability and remove temporary diagnostics.

- [x] Task: TDD - Verify all termination conditions (Pylon loss, Trainee army loss, Opponent army loss) in a unified test suite. df8d017
- [x] Task: Remove or toggle off diagnostic logging added in Phase 1. df8d017
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Regression & Validation' (Protocol in workflow.md)