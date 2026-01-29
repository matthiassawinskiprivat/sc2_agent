# Implementation Plan - Fix Unload AttributeError in SC2Bot

## Phase 1: Bug Fix Implementation [checkpoint: 0c184fc]
- [x] Task: Fix Unload Logic in `SC2Bot`. [0df303f]
    - [x] Create failing test `tests/test_sc2bot_unload_fix.py` to reproduce the `AttributeError`.
    - [x] Update `SC2Bot.apply_action` to use `AbilityId.UNLOADALL`.
- [ ] Task: Conductor - User Manual Verification 'Bug Fix Implementation' (Protocol in workflow.md)

## Phase 2: Integration Verification [checkpoint: 7d80dcf]
- [x] Task: Run integration test `tests/test_level_drop_system.py`. [9de9020]
    - [ ] Verify that the environment no longer crashes during unload.
- [ ] Task: Conductor - User Manual Verification 'Integration Verification' (Protocol in workflow.md)
