# Implementation Plan - Unify LevelDrop Configuration

## Phase 1: Configuration & Logic Refactoring

- [x] Task: Update `config/level_drop.yaml` [9804cee]
    - Move `rewards`, `active_layers`, and `action_heads` into the `env` section.
- [ ] Task: Create Refactoring Test Case
    - [ ] Create `tests/test_level_drop_config_refactor.py`
    - [ ] Test 1: Verify `LevelDropEnv` loads config correctly from the new structure.
    - [ ] Test 2: Verify `LevelDropBot` action space size matches `len(action_heads)`.
    - [ ] Test 3: Verify `LevelDropBot.action_masks` shape matches the dynamic config.
- [ ] Task: Refactor `LevelDropEnv`
    - [ ] Update `__init__` to read `action_heads` from `env` config.
    - [ ] Update `action_space` initialization to use `len(action_heads)`.
- [ ] Task: Refactor `LevelDropBot`
    - [ ] Update `__init__` to parse `action_heads`.
    - [ ] Update `apply_action` to map head indices to action strings dynamically.
    - [ ] Update `action_masks` to construct masks dynamically based on the list.
    - [ ] Update `calculate_drop_reward` to read from the nested `env.rewards` path.
- [ ] Task: Conductor - User Manual Verification 'Configuration & Logic Refactoring' (Protocol in workflow.md)
