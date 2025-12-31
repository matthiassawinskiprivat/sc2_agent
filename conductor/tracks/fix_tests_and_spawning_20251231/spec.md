# Specification: Fix Failing Tests and Level 3 Spawning

## Overview
This track addresses regressions introduced during a recent refactoring. It focuses on resolving 13 failing unit and integration tests and fixing the spawning logic in the Level 3 environment to respect the `targets` configuration parameter.

## Functional Requirements
### 1. Level 3 Environment Fixes
- **Spawning Logic:** Update `Level3Bot.reset_game_state` and `Level3Env` to correctly spawn the number of Zerglings and Beacons specified by the `targets` parameter in the configuration.
- **Configuration Passing:** Ensure `targets` (and other relevant environment parameters) are correctly passed from the YAML configuration to the `Level3Env` and subsequently to the `Level3Bot`.

### 2. Test Resolution (13 Failures)
- **Environment & Logic:**
    - Fix `test_coordinates_manual.py`: Resolve `AttributeError: 'Level0Env' object has no attribute 'agent'`.
    - Fix `test_config_reward.py`: Correct expected reward values (e.g., `-0.075` vs `-0.05`).
    - Fix `test_env_action_flow.py`: Ensure `client.actions` is called correctly during step execution in mocks.
- **Infrastructure & Utilities:**
    - Fix `test_model_utils.py`: Update `test_graft_model_weights_no_change_needed` to handle object identity vs. equality correctly.
    - Fix `test_sync_vec_env_logging.py`: Update logger mocks or implementation to ensure debug messages are captured correctly.
    - Fix `test_watchdog.py` and `test_watchdog_integration.py`: Investigate why the watchdog timer is not triggering callbacks in the test environment (possibly due to thread/loop issues).
- **Integration:**
    - Fix `test_training_pipeline_integration.py`: Resolve `ValueError` related to unpacking and `Race` enum validation in mocked scenarios.

## Non-Functional Requirements
- **TDD Adherence:** Every fix must be verified by making the corresponding test pass.
- **Minimal Changes:** Focus strictly on fixing the reported issues without introducing unrelated changes.

## Acceptance Criteria
- [ ] `pytest` execution results in 0 failures.
- [ ] Level 3 environment successfully spawns multiple units/beacons when `targets > 1`.
- [ ] `Level3Env` correctly cycles through all active units before stepping the game simulation.

## Out of Scope
- Implementing new features for Level 3 beyond the spawning fix.
- Refactoring the `ObservationTransformer` unless required for test fixes.
