# Specification: Hybrid Action Heads (Categorical Unload)

## Goal
Optimize the action space by moving non-spatial actions (starting with `unload`) from a 128x128 spatial grid to a single categorical "button" head. This simplifies the learning task and provides a template for future race-specific abilities (Stimpack, Siege, etc.).

## Requirements

### 1. Configuration (`src/utils/config_manager.py`)
- Implement `get_cumulative_categorical_heads(target_level: str)` following the additive inheritance pattern.
- Update `config/level_drop_0.yaml` to move `"unload"` from `action_heads` to `categorical_heads`.

### 2. Action Space (`train/envs/level_0_env.py`)
- Redefine `action_space = Discrete( (N_spatial * Grid^2) + N_categorical )`.
- Ensure the categorical heads are appended at the end of the action vector.

### 3. Bot Logic (`src/sc2_bot.py`)
- **Action Decoding:** Update `apply_action` to handle indices > total spatial area.
- **Race-Agnostic Unload:** Map the `"unload"` action string to:
  - `AbilityId.UNLOADALL` (Terran/Zerg/Generic)
  - `AbilityId.UNLOADALL_WARPPRISM` (Protoss)
- **Intelligent Masking:**
  - `unload` categorical bit is `True` ONLY IF:
    - Unit has `cargo_used > 0`.
    - Unit is over **pathable terrain** (check `pathing_grid` at current unit position).

### 4. Model Surgery (`src/utils/model_utils.py`)
- Update `graft_model_weights` to handle categorical expansion.
- When new categorical neurons are added:
  - Initialize their **bias to +2.0** ("Loud" initialization) to force exploration in pre-trained models.
  - Initialize weights to small random values.

### 5. Verification
- **Unit Test:** `tests/test_hybrid_action_space.py` to verify index decoding and mask alignment.
- **Integration Test:** Verify pathing-aware masking in a mock SC2 environment.
- **Grafting Test:** Verify that grafting a CTF model to the new space results in a +2.0 bias for the new `unload` slot.
