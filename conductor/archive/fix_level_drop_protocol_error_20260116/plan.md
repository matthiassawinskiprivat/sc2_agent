# Implementation Plan - Restore Game Creation with Dynamic Race Detection

## Phase 1: Dynamic Race Parsing
- [x] Task: Implement race parsing from map names in `LevelDropEnv`. [869d989]
    - [ ] Create failing test `tests/test_level_drop_race_parsing.py` to verify that races are correctly extracted from names like `...Terran_vs_Zerg`.
    - [ ] Add `_parse_races_from_map_name()` to `LevelDropEnv`.
- [x] Task: Conductor - User Manual Verification 'Dynamic Race Parsing' (Protocol in workflow.md) [checkpoint: 869d989]

## Phase 2: Lifecycle Restoration
- [x] Task: Restore `create_game` in `LevelDropEnv._setup_game_session`. [c79e35f]
    - [ ] Update `_setup_game_session` to use the parsed races when calling `self._controller.create_game`.
    - [ ] Ensure `join_game` uses the parsed agent race.
- [x] Task: Conductor - User Manual Verification 'Lifecycle Restoration' (Protocol in workflow.md) [checkpoint: c79e35f]

## Phase 3: Verification & Regression
- [x] Task: Run integration test `tests/test_level_drop_system.py`. [c79e35f]
    - [ ] Verify no `ProtocolError` occurs during reset.
    - [ ] Verify units are correctly controlable.
- [x] Task: Conductor - User Manual Verification 'Verification & Regression' (Protocol in workflow.md) [checkpoint: c79e35f]
