# Implementation Plan - Refactor Action Heads Configuration

## Phase 1: Core Logic (ConfigManager)

- [x] Task: Implement `get_cumulative_action_heads` in `ConfigManager` [206bc4a]
    - [x] Write unit tests in `tests/test_additive_config.py` for the new method, covering both pipeline levels and unknown/off-pipeline levels.
    - [x] Implement `get_cumulative_action_heads` in `src/utils/config_manager.py` using the additive accumulation logic.
- [x] Task: Conductor - User Manual Verification 'Core Logic' (Protocol in workflow.md) [checkpoint: 206bc4a]

## Phase 2: Configuration & Level 0 Refactoring

- [x] Task: Update Level 0 Configuration [21acaf7]
    - [x] Add `action_heads: ["move"]` to `config/level_0.yaml`.
- [x] Task: Refactor `Level0Env` and `Level0Bot` [21acaf7]
    - [x] Write tests verifying that `Level0Env` initializes its action space based on the configuration.
    - [x] Update `Level0Env.__init__` to use `self.config_manager.get_cumulative_action_heads`.
    - [x] Update `Level0Bot` to construct its action mask dynamically based on `self.action_heads`.
- [x] Task: Regression Test - Level 0 [21acaf7]
    - [x] Run existing tests for Level 0 (`tests/test_bot_init_final.py`, `tests/test_env_action_flow.py`) to ensure no regressions.
- [x] Task: Conductor - User Manual Verification 'Config & Level 0' (Protocol in workflow.md) [checkpoint: 21acaf7]

## Phase 3: Level 7 Refactoring

- [x] Task: Update Level 7 Configuration [4c3b14b]
    - [x] Add `action_heads: ["attack"]` to `config/level_7.yaml`.
- [x] Task: Refactor `Level7Env` and `Level7Bot` [4c3b14b]
    - [x] Write tests verifying Level 7 has both "move" and "attack" heads.
    - [x] Update `Level7Env` to initialize with the cumulative list.
    - [x] Update `Level7Bot.action_masks` to dynamically concatenate masks based on the `action_heads` list.
- [x] Task: Regression Test - Level 7 [4c3b14b]
    - [x] Run existing tests for Level 7 (`tests/test_level_7_combat_env_20260101`, `tests/test_level_7_mask_mismatch.py`) to ensure no regressions.
- [x] Task: Conductor - User Manual Verification 'Level 7' (Protocol in workflow.md) [checkpoint: 4c3b14b]

## Phase 4: Level Drop & System Verification

- [x] Task: Update Level Drop Configuration [278c5e0]
    - [x] Update `config/level_drop.yaml` to set `action_heads: ["load", "unload"]`.
- [x] Task: Verify System-wide Accumulation [278c5e0]
    - [x] Write a test case that instantiates `LevelDropEnv` and verifies it has 4 action heads (move, attack, load, unload).
    - [x] Verify `LevelDropBot` continues to function correctly with the dynamic mapping.
- [x] Task: Regression Test - All Environments [278c5e0]
    - [x] Run the full test suite (`pytest`) to ensure `ConfigManager` changes haven't negatively impacted other environments (like CTF) that might rely on shared logic.
- [x] Task: Conductor - User Manual Verification 'Level Drop & System Verification' (Protocol in workflow.md) [checkpoint: 278c5e0]
