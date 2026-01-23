# Implementation Plan: Hybrid Action Heads

## Phase 1: Infrastructure & Configuration
- [ ] Add `get_cumulative_categorical_heads` to `src/utils/config_manager.py`.
- [ ] Update `config/level_drop_0.yaml`:
  - Move `"unload"` to `categorical_heads`.
- [ ] Update `train/envs/level_0_env.py`:
  - Update `__init__` to load categorical heads.
  - Correct `action_space` calculation.

## Phase 2: Bot Logic & Masking
- [ ] Update `src/sc2_bot.py`:
  - Implement hybrid index decoding in `apply_action`.
  - Implement race-agnostic `unload` command mapping.
  - Implement pathing-aware `unload` masking.
- [ ] **Test:** Create `tests/test_hybrid_masking.py` to verify pathing/cargo logic.

## Phase 3: Model Surgery (The "Loud" Graft)
- [ ] Update `src/utils/model_utils.py`:
  - Detect categorical expansion.
  - Apply `+2.0` bias to new neurons.
- [ ] **Test:** Create `tests/test_loud_grafting.py`.

## Phase 4: Integration & Cleanup
- [ ] Verify `LevelDropEnv` integration.
- [ ] Update `PROJECT_SUMMARY.md` or `ARCHITECTURE.md` to reflect the new hybrid action system.
