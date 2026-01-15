# Implementation Plan - Fix LevelDrop Initialization and Objective Logic

## Phase 1: Environment & Spawning Correction
- [x] Task: Update `config/level_drop_0.yaml` to use `env_class: "LevelDropEnv"`. [623b9b6]
- [x] Task: Refactor `LevelDropBot.reset_game_state` to skip unit clearing. [0b85e08]
    - [x] Create failing test `tests/test_level_drop_init_logic.py` to verify that `LevelDropBot` does not attempt to clear the map during reset.
    - [x] Implement the override in `LevelDropBot.reset_game_state` to skip cleanup.
- [x] Task: Conductor - User Manual Verification 'Environment & Spawning Correction' (Protocol in workflow.md) [checkpoint: 0b85e08]

## Phase 2: Objective Tracking & Multi-Unit Control
- [x] Task: Implement enemy main structure detection logic. [0b85e08]
    - [x] Create logic in `LevelDropBot` to find and cache the enemy's `Command Center`, `Hatchery`, or `Nexus`.
- [x] Task: Enable multi-unit cycling (Marines + Medivacs). [0b85e08]
    - [x] Override `get_active_units` in `LevelDropBot` to include both `MARINE` and `MEDIVAC` tags.
- [x] Task: Implement "Always Visible" Objective Rendering. [0b85e08]
    - [x] Update `LevelDropBot.get_observation_for_unit` to populate `command_target` with the detected enemy structure position, even if it is in Fog of War.
- [x] Task: Conductor - User Manual Verification 'Objective Tracking & Multi-Unit Control' (Protocol in workflow.md) [checkpoint: 0b85e08]

## Phase 3: System Integration & Final Verification
- [ ] Task: Create comprehensive integration test `tests/test_level_drop_system.py`.
    - [ ] Verify the environment cycles through multiple unit types correctly.
    - [ ] Verify the `command_objective` layer contains the correct target coordinates.
- [ ] Task: Run the `train_pipeline` for a few steps to visually/logically confirm Marines and Medivacs are controlled sequentially.
- [ ] Task: Conductor - User Manual Verification 'System Integration & Final Verification' (Protocol in workflow.md)

