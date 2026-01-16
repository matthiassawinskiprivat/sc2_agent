# Implementation Plan - Fix Cross-Race Spatial Unload Mechanics

## Phase 1: Research & Test Infrastructure [checkpoint: a52d92c]
- [x] Task: Verify Ability IDs and Unit Type IDs for all transport units. [f58a076]
    - [ ] Confirm `UNLOADALLAT_MEDIVAC`, `UNLOADALLAT_OVERLORD`, and `UNLOADALLAT_WARPPRISM` availability in `burnysc2`.
    - [ ] Identify `UnitTypeId` for each transport to use in mapping logic.
- [x] Task: Set up multi-race unit mocking for TDD. [2478fa6]
    - [ ] Ensure `tests/test_sc2bot_unload_fix.py` or a new test can easily simulate different unit types.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Research & Test Infrastructure' (Protocol in workflow.md)

## Phase 2: Core Implementation (TDD) [checkpoint: 512b70b]
- [x] Task: Implement Spatial Unload for Medivacs. [319d12c]
    - [ ] Create failing test `tests/test_unload_spatial_medivac.py` that asserts `UNLOADALLAT_MEDIVAC` is called with `target_pos`.
    - [ ] Update `SC2Bot.apply_action` to use unit-specific mapping for Medivacs.
    - [ ] Verify test passes.
- [x] Task: Implement Multi-Race Support (Overlord, Warp Prism). [aac7fea]
    - [ ] Create failing test `tests/test_unload_multi_race.py` covering Zerg and Protoss transport scenarios.
    - [ ] Implement the mapping dictionary/logic in `SC2Bot.apply_action`.
    - [ ] Verify tests pass.
- [x] Task: Implement Fallback and Safety Logic. [57897b8]
    - [ ] Create failing test for unknown transport units or units with no cargo.
    - [ ] Ensure the bot falls back to a generic unload or gracefully skips the action.
    - [ ] Verify all tests pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Core Implementation' (Protocol in workflow.md)

## Phase 3: Final Integration & Quality Gate
- [ ] Task: Run full test suite and verify coverage.
    - [ ] Ensure no regressions in `LevelDropEnv` or other bots.
    - [ ] Verify that `unload` actions are no longer ignored in logs (via manual dry run if possible).
- [ ] Task: Code style and documentation.
    - [ ] Ensure new mapping logic is documented.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Integration & Quality Gate' (Protocol in workflow.md)
