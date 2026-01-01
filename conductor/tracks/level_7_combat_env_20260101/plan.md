# Plan: Level 7 Environment (Combat Basics)

**Goal:** Implement a new training environment "Level 7" that introduces basic combat mechanics. The agent controls a Marine (Terran) and must decide between moving and attacking an enemy Zergling (Zerg).

**Core Requirements:**
*   **Units:** Controlled Marine vs. Enemy Zergling.
*   **Visibility:** No Fog of War (Perfect Information).
*   **Action Space:** Flattened discrete space (`2 * GridSize`). Indices `0..N` for Move, `N..2N` for Attack.
*   **Observation:** Include Allied/Enemy Health layers.
*   **Rewards:** Damage dealt (small), Kill (large).
*   **Weight Transfer:** Modify `graft_model_weights` to explicitly preserve "Move" weights in the first half of the new action head.

## [x] Phase 1: Configuration
- [x] Create `config/level_7.yaml`. e2ed44c
    - [x] Clone from `level_3.yaml`.
    - [x] Update `env` (Marine vs Zergling, rewards).
    - [x] Update `observation` (Active layers: allied/enemy health).
    - [x] Update `training` (Steps, larger_units).

## [x] Phase 2: Bot Logic (Level7Bot)
- [x] Create `src/level_7_bot.py`. 7311c47
    - [x] Inherit from `Level3Bot` (reusing unit cycling/assignment).
    - [x] Update `reset_game_state` for Marine/Zergling spawning.
    - [x] Override `apply_action` to handle Move (0..N) vs Attack (N..2N).
    - [x] Implement `calculate_combat_reward` (tracking health deltas).

## [x] Phase 3: Environment Implementation [checkpoint: 22d6e2b]
- [x] Create `train/envs/level_7_env.py`. cf4f1a6
    - [x] Inherit from `Level3Env`.
    - [x] Disable fog of war in `reset` (`debug_show_map`).
    - [x] Integrate `Level7Bot`.
    - [x] Update reward calculation to include combat rewards.

## [x] Phase 4: Integration
- [x] Update `train/envs/__init__.py` to export `Level7Env`. 73c9210
- [x] Update `train/train_pipeline.py` to register `Level7Env` in `ENV_REGISTRY`. 73c9210
- [x] **Crucial:** Modify `src/utils/model_utils.py` -> `graft_model_weights`. 73c9210
    - [x] Add logic: If `new_action_dim == 2 * old_action_dim`, copy old weights to `[:old_dim]` and initialize `[old_dim:]` randomly.

## [x] Phase 5: Testing [checkpoint: 26d4e53]
- [x] Create `tests/test_level_7_env.py`. 827a93b
    - [x] Test 1: Initialization (Units spawned correct types).
    - [x] Test 2: Action parsing (Move vs Attack commands sent).
    - [x] Test 3: Reward generation (Damage gives reward).
- [x] Create `tests/test_level_7_grafting.py`. 73c9210
    - [x] Verify that after grafting, the first half of the action head matches the old model's weights exactly (or interpolated equivalent).

## [~] Phase 6: Pipeline
- [~] Update `config/pipeline.yaml` to include `level_7`.
