# Plan: Fix Property Shadowing and Action Masking in Universal Agent

## Objective
Fix the reward drop from 20 to -20 in `level_move_multi_0` by removing property shadowing in bot classes. Also improve action masking for the Universal architecture.

## Key Files & Context
- `src/sc2_bot.py`: Base bot class shadowing `BotAI` properties.
- `src/sc2_bot_universal.py`: Universal bot class shadowing properties and having placeholder action masks.
- `src/agent/ctf_bot.py`: CTF bot class shadowing properties.
- `src/level_3_bot.py`: Level 3 bot bypassing curriculum and targeting logic for Universal architecture.

## Implementation Steps

### 1. Remove Property Shadowing
- In `src/sc2_bot.py`:
    - Remove initialization of `self.units`, `self.structures`, `self.enemy_units`, `self.enemy_structures`, and `self.all_units` in `__init__`.
    - Remove the same initializations in `reset_game_state`.
- In `src/sc2_bot_universal.py`:
    - Remove initialization of `self.units`, `self.structures`, `self.enemy_units`, `self.enemy_structures`, and `self.all_units` in `__init__`.
    - Remove the same initializations in `reset_game_state`.
- In `src/agent/ctf_bot.py`:
    - Remove initialization of `self.units`, `self.structures`, `self.enemy_units`, `self.enemy_structures`, and `self.all_units` in `__init__`.

### 2. Improve Universal Action Masking
- In `src/sc2_bot_universal.py`:
    - Update `action_masks` to use `agent_system` to calculate pathable movement masks.
    - Incorporate guidance curriculum if enabled in `training_conf`.
- In `src/level_3_bot.py`:
    - Update `action_masks` for `arch == "universal"` to still perform unit-to-beacon assignment so that `self.beacon` is set correctly before calling the Universal masking logic.

### 3. Verification
- Create `tests/test_bot_init_final.py` to verify that `units` is a property and `all_units` is updated by `_prepare_step`.
- Run `repro_level_3_reward.py` (updated) to ensure positive rewards.
- Run existing tests to ensure no regressions.

## Verification & Testing
- `python tests/test_bot_init_final.py`
- `python repro_level_3_reward.py`
- `pytest tests/test_action_mask_shape.py`
- `pytest tests/test_action_mask_validity.py`
