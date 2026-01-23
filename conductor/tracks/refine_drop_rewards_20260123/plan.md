# Implementation Plan: Drop Level Reward Refinement

## Phase 1: Environment Logic & Tracking [checkpoint: 6b1e7ec]
This phase focuses on updating the internal state tracking of the agent/environment to monitor load/unload events and unit health.

- [x] Task: Implement Load/Unload Registry (9d05ec4)
    - [x] Write unit tests for `LevelDropBot` state tracking (initializing and updating registries).
    - [x] Update `LevelDropBot` to maintain `initially_loaded_units` and `target_unloaded_units` sets.
- [x] Task: Implement Unit Health & Shield Utility (44d9772)
    - [x] Write unit tests for a utility that calculates total effective health (Health + Shield).
    - [x] Implement/Update unit health check logic in `LevelDropBot`.
- [x] Task: Refactor Loading Reward Calculation (6acb18d)
    - [x] Write tests for the scaled loading reward formula (including cargo multiplier).
    - [x] Write tests for the "Emergency Pick-up" (damaged unit) condition.
    - [x] Implement the refined loading reward logic in `LevelDropBot`.
- [x] Task: Refactor Unloading Reward Calculation (c3ab587)
    - [x] Write tests for the scaled unloading reward formula (decreasing with each unit).
    - [x] Implement the refined unloading reward logic in `LevelDropBot`.
- [x] Task: Conductor - User Manual Verification 'Environment Logic & Tracking' (Protocol in workflow.md)

## Phase 2: Configuration & Integration [checkpoint: cc106f8]
Updating the YAML configurations to reflect the new reward structure and verifying the end-to-end signal flow.

- [x] Task: Update Configuration Files (1af667c)
- [x] Task: Integration Test for Reward Signals (3fcbb8f)
    - [x] Create a specialized integration test that mocks a loading sequence and verifies the exact reward values returned by the environment.
- [x] Task: Conductor - User Manual Verification 'Configuration & Integration' (Protocol in workflow.md)