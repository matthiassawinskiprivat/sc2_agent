# Plan: Dynamic Unit Costs and Budgeted CTF Spawning

## Phase 1: Unit Data Extraction Tool Update [checkpoint: 0d09dae]
- [x] Task: Create tests for unit data loading to verify `data/unit_data.json` integrity. (03b0bc6)
- [x] Task: Refactor `generate_unit_json.py` to use `burnysc2` for extracting unit stats (costs, stats, supply). (03b0bc6)
- [x] Task: Execute `generate_unit_json.py` and verify the generated `data/unit_data.json` contains accurate data for the required unit pool. (03b0bc6)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Unit Data Extraction Tool Update' (Protocol in workflow.md)

## Phase 2: Budgeted Spawning Implementation
- [~] Task: Create unit tests for the budgeted spawning logic (mocking unit costs and budgets).
- [ ] Task: Update `CaptureTheFlagSingleEnv` configuration to support `mineral_budget` and `vespene_budget`.
- [ ] Task: Refactor `_spawn_initial_units` in `CaptureTheFlagSingleEnv` to implement the budget-based loop.
- [ ] Task: Ensure the `CaptureTheFlagBot` correctly identifies and handles the new unit types from the `unit_pool`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Budgeted Spawning Implementation' (Protocol in workflow.md)

## Phase 3: Integration and Validation
- [ ] Task: Create integration tests to verify spawning composition across different budgets and unit pools.
- [ ] Task: Verify that units like `REAPER`, `ZEALOT`, and `ROACH` are correctly controlled and interact in the environment.
- [ ] Task: Run a stability check (10-20 resets) to ensure budget consistency and no spawning crashes.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Validation' (Protocol in workflow.md)
