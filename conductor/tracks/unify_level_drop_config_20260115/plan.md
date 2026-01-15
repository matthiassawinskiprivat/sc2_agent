# Implementation Plan - Unify LevelDrop Configuration

## Phase 1: Configuration & Logic Refactoring

- [x] Task: Update `config/level_drop.yaml` [9804cee]
    - Move `rewards`, `active_layers`, and `action_heads` into the `env` section.
- [x] Task: Create Refactoring Test Case [9c938e6]
    - [x] Create `tests/test_level_drop_config_refactor.py`
    - [x] Test 1: Verify `LevelDropEnv` loads config correctly from the new structure.
    - [x] Test 2: Verify `LevelDropBot` action space size matches `len(action_heads)`.
    - [x] Test 3: Verify `LevelDropBot.action_masks` shape matches the dynamic config.
- [x] Task: Refactor `LevelDropEnv` [318691a]
    - [x] Update `__init__` to read `action_heads` from `env` config.
    - [x] Update `action_space` initialization to use `len(action_heads)`.
- [x] Task: Refactor `LevelDropBot` [318691a]
    - [x] Update `__init__` to parse `action_heads`.
    - [x] Update `apply_action` to map head indices to action strings dynamically.
    - [x] Update `action_masks` to construct masks dynamically based on the list.
    - [x] Update `calculate_drop_reward` to read from the nested `env.rewards` path.
- [x] Task: Conductor - User Manual Verification 'Configuration & Logic Refactoring' (Protocol in workflow.md) [checkpoint: fee1c98]
