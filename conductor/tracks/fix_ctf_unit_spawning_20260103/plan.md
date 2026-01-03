# Plan - Fix Capture the Flag Unit Spawning

This plan addresses the configuration loading mismatch and unit cost discrepancies in the CTF environments, ensuring that the full `unit_pool` is correctly utilized during spawning.

## Phase 1: Configuration Standardization

- [x] Task: Update `config/capture_the_flag.yaml` to use `mineral_budget` and `vespene_budget` instead of `minerals` and `vespene`. (f73986e)
- [x] Task: Audit other CTF-related YAML configs (e.g., `test_ctf_pipeline.yaml`) for similar discrepancies and update them. (891ce4e)
- [x] Task: Update `tests/reproduce_unit_pool_bug.py` to use the new standardized keys in its test cases. (20eaec0)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration Standardization' (Protocol in workflow.md) (20eaec0)

## Phase 2: Environment Logic Updates

- [x] Task: Update `train/envs/capture_the_flag_single_env.py` to robustly load `mineral_budget`, `vespene_budget`, and `unit_pool` from the flattened `training_conf`. (3696507)
- [x] Task: Update `train/envs/capture_the_flag_env.py` to:
    - Load unit costs from `data/unit_data.json` instead of hardcoded `UNIT_COSTS`.
    - Robustly load parameters from `training_conf` using standardized keys.
    - Synchronize its `_calculate_spawns` logic with the single-env version for consistency. (214d573)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Environment Logic Updates' (Protocol in workflow.md) (214d573)

## Phase 3: Verification and Regression Testing

- [x] Task: Run `tests/reproduce_unit_pool_bug.py` and confirm all tests pass. (214d573)
- [x] Task: Run `tests/integration_test_ctf_unit_composition.py` to ensure real game spawning works for all units in the pool. (214d573)
- [x] Task: Run existing environment tests (`tests/test_capture_the_flag_env.py`, `tests/test_capture_the_flag_single_env.py`) to ensure no regressions. (214d573)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Verification and Regression Testing' (Protocol in workflow.md) (214d573)
