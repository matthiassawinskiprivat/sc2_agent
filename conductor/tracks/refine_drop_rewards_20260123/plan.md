# Implementation Plan: Drop Level Reward Refinement

## Phase 1: Environment Logic & Tracking
This phase focuses on updating the internal state tracking of the agent/environment to monitor load/unload events and unit health.

- [x] Task: Implement Load/Unload Registry (9d05ec4)
    - [x] Write unit tests for `LevelDropBot` state tracking (initializing and updating registries).
    - [x] Update `LevelDropBot` to maintain `initially_loaded_units` and `target_unloaded_units` sets.
- [x] Task: Implement Unit Health & Shield Utility (44d9772)
    - [x] Write unit tests for a utility that calculates total effective health (Health + Shield).
    - [x] Implement/Update unit health check logic in `LevelDropBot`.
- [x] Task: Refactor Loading Reward Calculation (7e3a9c2)
    - [x] Write tests for the scaled loading reward formula (including cargo multiplier).
    - [x] Write tests for the "Emergency Pick-up" (damaged unit) condition.
    - [x] Implement the refined loading reward logic in `LevelDropBot`.
- [ ] Task: Refactor Unloading Reward Calculation
    - [ ] Write tests for the scaled unloading reward formula (decreasing with each unit).
    - [ ] Implement the refined unloading reward logic in `LevelDropBot`.
- [ ] Task: Conductor - User Manual Verification 'Environment Logic & Tracking' (Protocol in workflow.md)

## Phase 2: Configuration & Integration
Updating the YAML configurations to reflect the new reward structure and verifying the end-to-end signal flow.

- [ ] Task: Update Configuration Files
    - [ ] Update `config/level_drop_0.yaml` (and related drop configs) to include `base_load_reward` and `base_unload_reward`.
- [ ] Task: Integration Test for Reward Signals
    - [ ] Create a specialized integration test that mocks a loading sequence and verifies the exact reward values returned by the environment.
- [ ] Task: Conductor - User Manual Verification 'Configuration & Integration' (Protocol in workflow.md)
