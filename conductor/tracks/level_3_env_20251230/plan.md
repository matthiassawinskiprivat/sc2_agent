# Implementation Plan: Level 3 Environment (Multi-Zergling Control)

This plan outlines the steps to implement `Level3Env`, enabling multi-agent control for sequential beacon collection, and updating the pipeline to support it.

## Phase 1: Infrastructure & Configurable Pipeline
- [x] Task: Extend `train_pipeline.py` to support dynamic environment class selection. It should check for an `env_class` parameter in the YAML config or use a mapping to decide which class to instantiate (defaulting to `Level0Env`).
- [x] Task: Create `train/envs/level_3_env.py` and implement basic inheritance from `Level0Env`.
- [x] Task: Ensure `Level3Env` correctly loads the `targets` parameter from `config/level_3.yaml`.
- [x] Task: Create a base test file `tests/test_level_3_env.py` that verifies the environment can be instantiated and initialized via the updated pipeline logic.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure' (Protocol in workflow.md)

## Phase 2: Multi-Entity Spawning & Removal Logic
- [x] Task: Override `_setup_game_session` in `Level3Env` to spawn `N` zerglings and `N` beacons based on the `targets` config.
- [x] Task: Implement TDD for entity spawning: Test that `reset()` results in the correct number of units on the map.
- [x] Task: Implement the "Collection" logic: Detect when a specific zergling reaches a specific beacon and remove both from the active state.
- [x] Task: Implement TDD for removal logic: Test that contact between a zergling and beacon reduces the active unit count.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Multi-Entity Logic' (Protocol in workflow.md)

## Phase 3: Sequential Step Loop & Reward Normalization
- [x] Task: Override `_async_step` (or equivalent) to implement the sequential control loop (one forward pass per active zergling).
- [x] Task: Update the reward calculation to average the distance penalty and normalize the success reward by `targets`.
- [x] Task: Implement TDD for rewards: Verify that reward values match the averaged/normalized specification.
- [x] Task: Implement TDD for sequential control: Verify that distinct actions move different zerglings in a single step.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Control & Rewards' (Protocol in workflow.md)

## Phase 4: Integration & Final Verification
- [x] Task: Verify that `Level3Env` works with the existing `train_pipeline.py` using the new dynamic loading.
- [x] Task: Run regression tests on `Level0Env` to ensure no behavior has changed.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Integration' (Protocol in workflow.md)
