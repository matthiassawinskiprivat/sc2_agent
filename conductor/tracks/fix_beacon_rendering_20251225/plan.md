# Plan: Fix Beacon Rendering in Observation

## Phase 1: Diagnosis and Reproduction [checkpoint: 69ab492]
- [x] Task: Create a failing test case `tests/test_beacon_visibility.py` that replicates the training environment setup and asserts the Beacon is present on the `ENEMY_UNIT_DENSITY` layer. 69ab492
- [x] Task: Inspect `src/transformers/observation_transformer.py` to identify why the `render_beacon` flag or rendering logic might be failing. 69ab492
- [x] Task: Audit the configuration flow from `config/pipeline.yaml` (or level-specific configs) through `train_pipeline.py` to the environment and transformer. 69ab492
- [x] Task: Conductor - User Manual Verification 'Phase 1: Diagnosis and Reproduction' (Protocol in workflow.md) 69ab492

## Phase 2: Implementation [checkpoint: 69ab492]
- [x] Task: Fix the `ObservationTransformer` rendering logic to ensure `UnitTypeId.BEACON_TERRANSMALL` is drawn when `render_beacon` is enabled. 69ab492
- [x] Task: Correct any broken configuration propagation that prevents `render_beacon=True` from reaching the transformer during training. 69ab492
- [x] Task: Verify the fix using the `tests/test_beacon_visibility.py` regression test (Green Phase). 69ab492
- [x] Task: Conductor - User Manual Verification 'Phase 2: Implementation' (Protocol in workflow.md) 69ab492

## Phase 3: Validation and Quality Assurance [checkpoint: b888c8d]
- [x] Task: Run a short training run (approx. 5-10 minutes) using `debug_train.py` or similar to confirm the agent's reward improves or movement becomes directed. b888c8d
- [x] Task: Execute the full test suite `pytest tests/` to ensure no regressions in other layers or agent behavior. b888c8d
- [x] Task: Conductor - User Manual Verification 'Phase 3: Validation and Quality Assurance' (Protocol in workflow.md) b888c8d
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Validation and Quality Assurance' (Protocol in workflow.md)
