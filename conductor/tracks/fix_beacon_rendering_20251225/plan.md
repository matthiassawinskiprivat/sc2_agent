# Plan: Fix Beacon Rendering in Observation

## Phase 1: Diagnosis and Reproduction
- [x] Task: Create a failing test case `tests/test_beacon_visibility.py` that replicates the training environment setup and asserts the Beacon is present on the `ENEMY_UNIT_DENSITY` layer.
- [x] Task: Inspect `src/transformers/observation_transformer.py` to identify why the `render_beacon` flag or rendering logic might be failing.
- [x] Task: Audit the configuration flow from `config/pipeline.yaml` (or level-specific configs) through `train_pipeline.py` to the environment and transformer.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Diagnosis and Reproduction' (Protocol in workflow.md)

## Phase 2: Implementation
- [ ] Task: Fix the `ObservationTransformer` rendering logic to ensure `UnitTypeId.BEACON_TERRANSMALL` is drawn when `render_beacon` is enabled.
- [ ] Task: Correct any broken configuration propagation that prevents `render_beacon=True` from reaching the transformer during training.
- [ ] Task: Verify the fix using the `tests/test_beacon_visibility.py` regression test (Green Phase).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md)

## Phase 3: Validation and Quality Assurance
- [ ] Task: Run a short training run (approx. 5-10 minutes) using `debug_train.py` or similar to confirm the agent's reward improves or movement becomes directed.
- [ ] Task: Execute the full test suite `pytest tests/` to ensure no regressions in other layers or agent behavior.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Validation and Quality Assurance' (Protocol in workflow.md)
