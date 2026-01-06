# Plan: Restore Bot Attributes and Fix Final Termination

This plan restores library compatibility for `CaptureTheFlagBot` and implements a robust, Level 7-inspired termination check.

## Phase 1: Restore BotAI Compatibility
Restore the internal variables and structures required by the StarCraft II library.

- [x] Task: TDD - Create a test verifying that `CaptureTheFlagBot` can undergo a `_prepare_step` (mocked) without `AttributeError`. f4c508e
- [x] Task: Restore `_initialize_variables()` and remove manual attribute deletions in `CaptureTheFlagBot.__init__`. f4c508e
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Restore BotAI Compatibility' (Protocol in workflow.md)

## Phase 2: Robust Termination Implementation
Implement the Level 7-style unit filtering and asymmetric termination logic.

- [x] Task: TDD - Update `_has_combat_units` tests to verify detection using the library's `Units` filtering approach. f4c508e
- [x] Task: Implement robust `_has_combat_units` in `CaptureTheFlagEnv` using `bot.units.filter(...)`. f4c508e
- [x] Task: Ensure `_check_termination` correctly applies the asymmetric logic (Trainee loss = End). f4c508e
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Robust Termination Implementation' (Protocol in workflow.md)

## Phase 3: Integration & Final Test
Verify the fix across all CTF environments and ensure the pipeline starts correctly.

- [x] Task: Integration Test - Run `tests/test_ctf_asymmetric_integration_fix.py` to ensure no startup crashes. f4c508e
- [x] Task: Verify termination flow in `CaptureTheFlagSingleEnv` matches the multi-client version. f4c508e
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Final Test' (Protocol in workflow.md)